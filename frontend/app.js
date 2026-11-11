const API = "/api/v1";
const LABELS = [
  ["R01", "Đường/cầu/hạ tầng hư hỏng"], ["R02", "Điện lực/chiếu sáng"],
  ["R03", "Cống/hố ga/thoát nước"], ["R04", "Vỉa hè/lòng đường"],
  ["R05", "Cây xanh/vật cản"], ["R06", "Biển báo"], ["R07", "Tổ chức giao thông"],
  ["R08", "Đèn tín hiệu"], ["R09", "Rác thải/vệ sinh môi trường"],
  ["R10", "Đỗ xe cản trở"], ["R11", "Khác"],
];
const byId = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[char]));

let reportMap = null;
let reportMarker = null;
let adminMap = null;
let adminMarkers = null;

function setMessage(element, message, type = "") {
  element.textContent = message;
  element.className = type ? `message ${type}` : "message";
}

function setCoordinates(latitude, longitude) {
  byId("latitude").value = Number(latitude).toFixed(7);
  byId("longitude").value = Number(longitude).toFixed(7);
  if (reportMap) {
    const point = [Number(latitude), Number(longitude)];
    if (reportMarker) reportMarker.setLatLng(point);
    else reportMarker = L.marker(point).addTo(reportMap);
    reportMap.setView(point, 16);
  }
}

function initializeReportMap() {
  const container = byId("report-map");
  if (!window.L || !container) {
    container.textContent = "Không tải được thư viện bản đồ. Có thể nhập tọa độ thủ công.";
    return;
  }
  reportMap = L.map(container).setView([16.0471, 108.2068], 13);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(reportMap);
  reportMap.on("click", (event) => setCoordinates(event.latlng.lat, event.latlng.lng));
  setTimeout(() => reportMap.invalidateSize(), 100);
}

initializeReportMap();

byId("locate-button").addEventListener("click", () => {
  if (!navigator.geolocation) {
    setMessage(byId("form-message"), "Trình duyệt không hỗ trợ định vị.", "error");
    return;
  }
  navigator.geolocation.getCurrentPosition(
    (position) => setCoordinates(position.coords.latitude, position.coords.longitude),
    (error) => setMessage(byId("form-message"), `Không lấy được vị trí: ${error.message}`, "error"),
    { enableHighAccuracy: true, timeout: 10000 },
  );
});

byId("report-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = byId("submit-report");
  const message = byId("form-message");
  const text = byId("report-text").value.trim();
  const latitudeValue = byId("latitude").value;
  const longitudeValue = byId("longitude").value;
  if (Boolean(latitudeValue) !== Boolean(longitudeValue)) {
    setMessage(message, "Vui lòng nhập cả vĩ độ và kinh độ, hoặc để trống cả hai.", "error");
    return;
  }

  const payload = { text, include_probabilities: true };
  if (latitudeValue && longitudeValue) {
    payload.latitude = Number(latitudeValue);
    payload.longitude = Number(longitudeValue);
  }

  button.disabled = true;
  button.querySelector("span").textContent = "Đang xử lý...";
  setMessage(message, "");
  try {
    const imageFile = byId("image-file").files[0];
    if (imageFile) {
      const formData = new FormData();
      formData.append("file", imageFile);
      const uploadResponse = await fetch(`${API}/uploads/image`, { method: "POST", body: formData });
      const uploadBody = await uploadResponse.json();
      if (!uploadResponse.ok) throw new Error(uploadBody.detail || "Không thể tải ảnh.");
      payload.image_url = uploadBody.image_url;
    }

    const response = await fetch(`${API}/reports`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Không thể gửi phản ánh.");
    renderResult(body);
    setMessage(message, "Phản ánh đã được tiếp nhận.", "success");
  } catch (error) {
    setMessage(message, error.message, "error");
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = "Gửi phản ánh";
  }
});

function renderResult(result) {
  byId("empty-result").classList.add("hidden");
  byId("prediction-result").classList.remove("hidden");
  byId("predicted-name").textContent = result.predicted_category.name;
  const percent = Math.max(0, Math.min(100, result.confidence * 100));
  byId("confidence-value").textContent = `${percent.toFixed(1)}%`;
  byId("confidence-bar").style.width = `${percent}%`;
  byId("report-code").textContent = result.report_code;
  byId("report-status").textContent = result.status;
  byId("review-warning").classList.toggle("hidden", !result.needs_review);
}

function adminHeaders() {
  return { "X-Admin-Key": byId("admin-key").value.trim() };
}

async function adminFetch(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: { ...adminHeaders(), ...(options.headers || {}) },
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || "Yêu cầu quản trị thất bại.");
  return body;
}

byId("load-admin").addEventListener("click", loadAdmin);

async function loadAdmin() {
  const message = byId("admin-message");
  if (!byId("admin-key").value.trim()) {
    setMessage(message, "Hãy nhập Admin API key.", "error");
    return;
  }
  try {
    const [stats, reports] = await Promise.all([
      adminFetch("/admin/stats"),
      adminFetch("/admin/reports?skip=0&limit=100"),
    ]);
    renderStats(stats);
    renderAdminReports(reports.items);
    renderAdminMap(reports.items);
    setMessage(message, `Đã tải ${reports.items.length} / ${reports.total} phản ánh.`, "success");
  } catch (error) {
    setMessage(message, error.message, "error");
  }
}

function renderStats(stats) {
  const cards = [
    ["Tổng phản ánh", stats.total_reports],
    ["Cần kiểm duyệt", stats.needs_review_predictions],
    ["Danh mục", stats.predictions_by_category.length],
  ];
  byId("admin-stats").innerHTML = cards.map(([label, value]) =>
    `<div class="stat-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`
  ).join("");
}

function renderAdminReports(items) {
  const table = byId("reports-table");
  if (!items.length) {
    table.innerHTML = '<tr><td colspan="6" class="empty-table">Chưa có phản ánh.</td></tr>';
    return;
  }
  table.innerHTML = items.map((item) => {
    const predicted = item.predicted_category?.name || "—";
    const confidence = item.confidence == null ? "—" : `${(item.confidence * 100).toFixed(1)}%`;
    const review = item.needs_review;
    const options = LABELS.map(([code, name]) =>
      `<option value="${code}" ${item.confirmed_category?.code === code ? "selected" : ""}>${code} — ${escapeHtml(name)}</option>`
    ).join("");
    return `<tr>
      <td><strong>${escapeHtml(item.report_code)}</strong></td>
      <td>${escapeHtml(item.text)}</td>
      <td>${escapeHtml(predicted)}</td>
      <td>${escapeHtml(confidence)}</td>
      <td><span class="status-pill ${review ? "review" : ""}">${escapeHtml(item.status)}</span></td>
      <td><select id="category-${item.id}" aria-label="Nhãn xác nhận">${options}</select>
        <button class="small-button" data-moderate="${item.id}">Lưu nhãn</button></td>
    </tr>`;
  }).join("");
}

function renderAdminMap(items) {
  const mapped = items.filter((item) => item.latitude != null && item.longitude != null);
  const container = byId("admin-map");
  if (!window.L || !container) {
    container.classList.add("hidden");
    return;
  }
  container.classList.toggle("hidden", mapped.length === 0);
  if (!mapped.length) return;
  if (!adminMap) {
    adminMap = L.map(container).setView([mapped[0].latitude, mapped[0].longitude], 12);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }).addTo(adminMap);
    adminMarkers = L.layerGroup().addTo(adminMap);
  }
  adminMarkers.clearLayers();
  const bounds = [];
  mapped.forEach((item) => {
    const point = [item.latitude, item.longitude];
    bounds.push(point);
    const title = `${escapeHtml(item.report_code)} — ${escapeHtml(item.predicted_category?.name || "Chưa phân loại")}`;
    L.marker(point).bindPopup(title).addTo(adminMarkers);
  });
  if (bounds.length === 1) adminMap.setView(bounds[0], 15);
  else adminMap.fitBounds(bounds, { padding: [20, 20], maxZoom: 15 });
  setTimeout(() => adminMap.invalidateSize(), 100);
}

byId("reports-table").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-moderate]");
  if (!button) return;
  const reportId = button.dataset.moderate;
  const categoryCode = byId(`category-${reportId}`).value;
  button.disabled = true;
  try {
    await adminFetch(`/admin/reports/${reportId}/moderation`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ category_code: categoryCode, note: "Cán bộ xác nhận từ Admin Dashboard demo." }),
    });
    setMessage(byId("admin-message"), "Đã lưu nhãn kiểm duyệt và nhật ký.", "success");
    await loadAdmin();
  } catch (error) {
    setMessage(byId("admin-message"), error.message, "error");
  } finally {
    button.disabled = false;
  }
});
