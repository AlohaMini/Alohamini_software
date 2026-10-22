import threading
import time
import unittest
from unittest.mock import patch

from alohamini.errors import ProtocolError, ResponseTimeoutError
from alohamini.remote.link import LatestLink
from alohamini.remote.protocol import decode_message, encode_message

try:
    import zmq
except ImportError:
    zmq = None

JPEG = b"\xff\xd8\x00unchanged wire image\xff\xd9"


def wait_for(check, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.002)
    raise AssertionError("condition did not become true before deadline")


@unittest.skipIf(zmq is None, "pyzmq is not installed")
class LatestLinkTests(unittest.TestCase):
    def setUp(self):
        self.context = zmq.Context()
        self.peer = self.context.socket(zmq.ROUTER)
        self.peer.setsockopt(zmq.LINGER, 0)
        port = self.peer.bind_to_random_port("tcp://127.0.0.1")
        self.endpoint = f"tcp://127.0.0.1:{port}"
        self.links = []

    def tearDown(self):
        for link in self.links:
            link.close()
        self.peer.close(linger=0)
        self.context.term()

    def link(self, timeout=0.5):
        link = LatestLink(self.endpoint, timeout_s=timeout).start()
        self.links.append(link)
        return link

    def receive(self):
        self.assertTrue(self.peer.poll(1000))
        identity, *parts = self.peer.recv_multipart()
        return identity, decode_message(parts)

    def reply(self, identity, request):
        kind, body, images = request
        self.peer.send_multipart([identity, *encode_message(kind, body, images)])

    def test_nonblocking_round_trip_keeps_original_jpeg_and_owns_submitted_body(self):
        link = self.link()
        body = {"id": 1, "nested": {"value": 2}}
        started = time.monotonic()
        link.submit("observation", body, {"front": JPEG})
        self.assertIsNone(link.poll())
        self.assertLess(time.monotonic() - started, 0.2)
        body["nested"]["value"] = 99
        identity, request = self.receive()
        self.assertEqual(request[1]["nested"]["value"], 2)
        self.assertEqual(request[2], {"front": JPEG})
        self.reply(identity, request)
        self.assertEqual(wait_for(link.poll), request)
        self.assertIsNone(link.poll())
        self.assertIsNone(link.error)

    def test_one_inflight_and_only_latest_pending_and_response_survive(self):
        link = self.link()
        link.submit("ping", {"id": 1})
        identity, first = self.receive()
        workers = [
            threading.Thread(target=link.submit, args=("ping", {"id": n})) for n in range(2, 12)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        link.submit("ping", {"id": 12})
        self.assertFalse(self.peer.poll(30), "pending requests must wait for the active response")
        self.reply(identity, first)
        identity, last = self.receive()
        self.assertEqual(last[1], {"id": 12})
        self.reply(identity, last)
        # Let the second response replace the unconsumed first response.
        wait_for(lambda: link._response is not None and link._response[1]["id"] == 12)
        self.assertEqual(link.poll(), last)
        self.assertFalse(self.peer.poll(30))

    def test_stalled_peer_times_out_without_replaying_pending_request(self):
        link = self.link(timeout=0.12)
        link.submit("observation", {"id": 1})
        identity, request = self.receive()
        link.submit("observation", {"id": 2})
        self.assertIsInstance(wait_for(lambda: link.error), ResponseTimeoutError)
        self.assertFalse(self.peer.poll(30))
        self.reply(identity, request)  # A late response cannot revive the failed worker.
        self.assertIsNone(link.poll())
        with self.assertRaisesRegex(RuntimeError, "failed"):
            link.submit("ping", {"id": 3})
        started = time.monotonic()
        link.close()
        link.close()
        self.assertLess(time.monotonic() - started, 0.3)
        self.assertFalse(link._thread.is_alive())

    def test_connection_wait_uses_the_same_bounded_request_deadline(self):
        self.peer.close(linger=0)
        link = self.link(timeout=0.08)
        link.submit("hello", {"id": 1})
        self.assertIsInstance(wait_for(lambda: link.error), ResponseTimeoutError)
        self.assertIsNone(link.poll())

    def test_close_interrupts_wait_without_socket_use_from_caller_thread(self):
        link = self.link(timeout=60)
        link.submit("ping", {"id": 1})
        self.receive()
        started = time.monotonic()
        link.close()
        self.assertLess(time.monotonic() - started, 0.3)
        self.assertFalse(link._thread.is_alive())
        self.assertIsNone(link.error)

    def test_malformed_or_excessive_response_ends_link_instead_of_draining(self):
        for parts in ([b"not JSON"], [b"{}", b"one", b"two"]):
            link = self.link()
            link.submit("ping", {"id": 1})
            identity, _ = self.receive()
            with patch("alohamini.remote.link.MAX_MESSAGE_FRAMES", 2):
                self.peer.send_multipart([identity, *parts])
                self.assertIsInstance(wait_for(lambda link=link: link.error), ProtocolError)
            self.assertIsNone(link.poll())


class LinkConfigurationTests(unittest.TestCase):
    def test_endpoint_and_timeout_validation_precedes_worker_creation(self):
        for endpoint in (
            "ipc:///tmp/remote",
            "http://localhost:9000",
            "tcp://*:9000",
            "tcp://localhost",
            "tcp://user@localhost:9000",
            "tcp://localhost:0",
            "tcp://localhost:65536",
            "tcp://localhost:9000/path",
            "tcp://localhost:9000?query=1",
            "tcp://localhost:9000\n",
            None,
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                LatestLink(endpoint)
        for timeout in (True, 0, -1, float("nan"), float("inf"), 61):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                LatestLink("tcp://127.0.0.1:9000", timeout_s=timeout)
        link = LatestLink("tcp://127.0.0.1:9000")
        with self.assertRaises(RuntimeError):
            link.submit("ping", {})
        link.close()
        link.close()
        with self.assertRaises(RuntimeError):
            link.start()


if __name__ == "__main__":
    unittest.main()
