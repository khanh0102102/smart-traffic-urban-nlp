(() => {
  "use strict";
  class ApiError extends Error {
    constructor(message, details = {}) {
      super(message);
      this.name = "ApiError";
      this.status = details.status ?? null;
      this.code = details.code ?? null;
      this.responseData = details.responseData ?? null;
      this.fieldErrors = details.fieldErrors ?? [];
      this.isNetworkError = Boolean(details.isNetworkError);
    }
  }
  function formatValidationDetail(detail) {
    if (typeof detail === "string" && detail.trim()) return detail;
    if (Array.isArray(detail)) return detail.map((item) => {
      const location = Array.isArray(item?.loc) ? item.loc.filter((part) => part !== "body").join(".") : "";
      const message = String(item?.msg || "Dữ liệu không hợp lệ");
      return location ? location + ": " + message : message;
    }).filter(Boolean).join("; ");
    if (detail && typeof detail === "object") return String(detail.message || detail.msg || "Yêu cầu không hợp lệ.");
    return "";
  }
  function normalizeError(error) {
    if (error instanceof ApiError) return error;
    const response = error?.response;
    const status = response?.status ?? null;
    const payload = response?.data ?? null;
    const detail = formatValidationDetail(payload?.detail);
    const fieldErrors = Array.isArray(payload?.detail) ? payload.detail : [];
    let message = detail;
    if (!response) {
      message = error?.code === "ECONNABORTED" || error?.code === "ETIMEDOUT"
        ? "Máy chủ phản hồi quá thời gian. Vui lòng thử lại."
        : "Không thể kết nối máy chủ. Kiểm tra kết nối mạng hoặc trạng thái backend.";
    } else if (!message) {
      const messages = {
        400:"Dữ liệu gửi lên không hợp lệ.",401:"Phiên đăng nhập không hợp lệ hoặc đã hết hạn.",
        403:"Tài khoản không có quyền thực hiện thao tác này.",404:"Không tìm thấy dữ liệu được yêu cầu.",
        409:"Dữ liệu bị xung đột. Hãy tải lại và thử lại.",413:"Tệp tải lên vượt quá giới hạn cho phép.",
        415:"Định dạng tệp không được hỗ trợ.",422:"Dữ liệu chưa đáp ứng quy tắc kiểm tra.",
        429:"Bạn thao tác quá nhanh. Vui lòng thử lại sau.",500:"Máy chủ gặp lỗi khi xử lý yêu cầu.",
        502:"Dịch vụ phía máy chủ hiện không khả dụng.",503:"Dịch vụ tạm thời chưa sẵn sàng."
      };
      message = messages[status] || (status >= 500 ? "Máy chủ gặp sự cố. Vui lòng thử lại sau." : "Yêu cầu không thành công.");
    }
    return new ApiError(message,{status,code:error?.code,responseData:payload,fieldErrors,isNetworkError:!response});
  }
  function createApiClient(options = {}) {
    if (!window.axios || typeof window.axios.create !== "function") throw new Error("Axios chưa được tải. Hãy kiểm tra CDN Axios trong trang HTML.");
    const getToken = typeof options.getToken === "function" ? options.getToken : () => null;
    const onUnauthorized = typeof options.onUnauthorized === "function" ? options.onUnauthorized : () => {};
    const instance = window.axios.create({
      baseURL:options.baseURL||"/api/v1",
      timeout:Number(options.timeout)>0?Number(options.timeout):25000,
      headers:{Accept:"application/json"}
    });
    instance.interceptors.request.use((config) => {
      config.headers = config.headers || {};
      if (config._skipAuth !== true) {
        const token = getToken();
        if (token) config.headers.Authorization = "Bearer " + token;
      }
      return config;
    });
    instance.interceptors.response.use(
      (response) => response.data,
      (error) => {
        const config = error?.config || {};
        if (error?.response?.status === 401 && config._skipAuth !== true && getToken()) onUnauthorized(normalizeError(error));
        return Promise.reject(normalizeError(error));
      }
    );
    return Object.freeze({
      request(path,config={}) { return instance.request({url:path,...config}); },
      get(path,config={}) { return instance.get(path,config); },
      post(path,data,config={}) { return instance.post(path,data,config); },
      patch(path,data,config={}) { return instance.patch(path,data,config); },
      delete(path,config={}) { return instance.delete(path,config); }
    });
  }
  function hasRole(user,allowedRoles) {
    return Boolean(user && typeof user.role==="string" && Array.isArray(allowedRoles) && allowedRoles.includes(user.role.toUpperCase()));
  }
  function assertRole(user,allowedRoles) {
    if (!hasRole(user,allowedRoles)) throw new ApiError("Tài khoản không có quyền truy cập khu vực này.",{status:403});
    return user;
  }
  window.STNLPApi=Object.freeze({ApiError,createApiClient,normalizeError,hasRole,assertRole});
})();