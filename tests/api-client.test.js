const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function loadApiClient() {
  let requestInterceptor;
  let responseSuccess;
  let responseFailure;
  let lastAxiosConfig;
  const fakeAxios = {
    create(config) {
      lastAxiosConfig = config;
      return {
        interceptors: {
          request: { use(handler) { requestInterceptor = handler; } },
          response: { use(success, failure) {
            responseSuccess = success;
            responseFailure = failure;
          } }
        },
        async request(config) {
          const prepared = await requestInterceptor({ headers: {}, ...config });
          return responseSuccess({ data: { ok: true, authorization: prepared.headers.Authorization || null }, config: prepared });
        },
        get(url, config) { return this.request({ ...config, method: "get", url }); },
        post(url, data, config) { return this.request({ ...config, method: "post", url, data }); },
        patch(url, data, config) { return this.request({ ...config, method: "patch", url, data }); },
        delete(url, config) { return this.request({ ...config, method: "delete", url }); }
      };
    }
  };
  const window = { axios: fakeAxios };
  const source = fs.readFileSync(path.join(__dirname, "../frontend/assets/js/api-client.js"), "utf8");
  vm.runInNewContext(source, { window, Promise, Object, Error, String, Boolean, Number, Array });
  return {
    api: window.STNLPApi,
    axiosConfig: () => lastAxiosConfig,
    reject: (error) => responseFailure(error)
  };
}

test("Axios client sets same-origin API base URL and attaches the Bearer token", async () => {
  const { api, axiosConfig } = loadApiClient();
  const client = api.createApiClient({ getToken: () => "jwt-citizen-token" });
  const response = await client.get("/reports/mine");
  assert.equal(axiosConfig().baseURL, "/api/v1");
  assert.equal(response.ok, true);
  assert.equal(response.authorization, "Bearer jwt-citizen-token");
});

test("login requests can explicitly skip the current Bearer token", async () => {
  const { api } = loadApiClient();
  const client = api.createApiClient({ getToken: () => "stale-token" });
  const response = await client.request("/auth/staff/login", { method: "post", _skipAuth: true });
  assert.equal(response.authorization, null);
});

test("HTTP 401 calls the portal unauthorized handler and normalizes the error", async () => {
  const { api, reject } = loadApiClient();
  let unauthorizedCalls = 0;
  const client = api.createApiClient({
    getToken: () => "expired-token",
    onUnauthorized: () => { unauthorizedCalls += 1; }
  });
  await client.get("/auth/me");
  const error = await reject({
    config: { _skipAuth: false },
    response: { status: 401, data: { detail: "Token expired" } }
  }).catch((value) => value);
  assert.equal(unauthorizedCalls, 1);
  assert.equal(error.status, 401);
  assert.equal(error.message, "Token expired");
});

test("HTTP 401 from a login request does not invalidate an existing session", async () => {
  const { api, reject } = loadApiClient();
  let unauthorizedCalls = 0;
  const client = api.createApiClient({
    getToken: () => "existing-token",
    onUnauthorized: () => { unauthorizedCalls += 1; }
  });
  await client.request("/auth/staff/login", { method: "post", _skipAuth: true });
  const error = await reject({
    config: { _skipAuth: true },
    response: { status: 401, data: { detail: "Invalid credentials" } }
  }).catch((value) => value);
  assert.equal(error.status, 401);
  assert.equal(unauthorizedCalls, 0);
});

test("FastAPI validation errors preserve field-level details", () => {
  const { api } = loadApiClient();
  const error = api.normalizeError({
    response: {
      status: 422,
      data: { detail: [{ loc: ["body", "raw_content"], msg: "Field required" }] }
    }
  });
  assert.equal(error.status, 422);
  assert.equal(error.fieldErrors.length, 1);
  assert.match(error.message, /raw_content: Field required/);
});

test("role guard allows only the explicitly supplied portal roles", () => {
  const { api } = loadApiClient();
  assert.equal(api.hasRole({ role: "CITIZEN" }, ["CITIZEN"]), true);
  assert.equal(api.hasRole({ role: "CITIZEN" }, ["ADMIN", "RECEIVER"]), false);
  assert.equal(api.assertRole({ role: "RECEIVER" }, ["ADMIN", "RECEIVER"]).role, "RECEIVER");
  assert.throws(
    () => api.assertRole({ role: "CITIZEN" }, ["ADMIN", "RECEIVER"]),
    (error) => error.status === 403
  );
});
