from och_annotate.baserow import BaserowClient


class FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.content = b"x" if payload is not None else b""

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        return self.responses.pop(0)


def _client(responses):
    client = BaserowClient("https://baserow.example.org", "tok")
    client.session = FakeSession(responses)
    return client


def test_iter_rows_follows_pagination():
    responses = [
        FakeResp({"results": [{"id": 1}, {"id": 2}], "next": "page2"}),
        FakeResp({"results": [{"id": 3}], "next": None}),
    ]
    client = _client(responses)
    rows = client.fetch_rows(1026)
    assert [r["id"] for r in rows] == [1, 2, 3]


def test_ensure_fields_only_creates_missing():
    responses = [
        FakeResp([{"name": "esmc_embedding"}]),  # list_fields
        FakeResp({"id": 99}),                    # create esmc_model
    ]
    client = _client(responses)
    created = client.ensure_fields(1026, ["esmc_embedding", "esmc_model"])
    assert created == {"esmc_embedding": False, "esmc_model": True}
    # exactly one POST (the missing field)
    posts = [c for c in client.session.calls if c[0] == "POST"]
    assert len(posts) == 1


def test_update_rows_chunks_batches():
    items = [{"id": i} for i in range(250)]
    responses = [FakeResp(None), FakeResp(None)]  # two chunks (200 + 50)
    client = _client(responses)
    client.update_rows(1026, items)
    patches = [c for c in client.session.calls if c[0] == "PATCH"]
    assert len(patches) == 2
