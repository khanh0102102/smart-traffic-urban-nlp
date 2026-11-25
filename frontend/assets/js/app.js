document.addEventListener("DOMContentLoaded", () => {
  const status = document.querySelector("[data-app-status]");
  if (status) {
    status.dataset.ready = "true";
  }
});
