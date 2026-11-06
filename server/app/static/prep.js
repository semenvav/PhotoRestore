// Редактор подготовки (cropper.js 1.x).
// Браузер только выбирает параметры; кроп и поворот на полном разрешении делает сервер.
// cropper.getData() отдаёт рамку в пикселях превью внутри описанного прямоугольника
// повёрнутого изображения; на сервер уходит то же самое в пикселях оригинала.
(() => {
  "use strict";

  const cfg = JSON.parse(document.getElementById("prep-config").textContent);
  const img = document.getElementById("prep-image");
  const angleInput = document.getElementById("angle");
  const angleOut = document.getElementById("angle-out");
  const resultSize = document.getElementById("result-size");
  const errorBox = document.getElementById("prep-error");
  const ratioButtons = document.querySelectorAll("[data-ratio]");

  let cropper = null;
  let scale = 1; // пиксели оригинала (после EXIF) / пиксели превью
  let rotate90 = 0;
  let angle = 0;
  let busy = false;

  const round1 = (v) => Math.round(v * 10) / 10;

  function totalRotation() {
    return rotate90 * 90 + angle;
  }

  function cropFull() {
    const c = cropper.getCanvasData();
    cropper.setData({ x: 0, y: 0, width: c.naturalWidth, height: c.naturalHeight });
  }

  function showAngle() {
    angleInput.value = String(angle);
    angleOut.textContent = `${angle > 0 ? "+" : ""}${angle.toFixed(1)}°`;
  }

  function showResult() {
    const d = cropper.getData();
    resultSize.textContent = `${Math.round(d.width * scale)} × ${Math.round(d.height * scale)} px`;
  }

  function applyRotation({ resetCrop }) {
    cropper.rotateTo(totalRotation());
    if (resetCrop) cropFull();
    showAngle();
    showResult();
  }

  function rotateQuarter(step) {
    rotate90 = (rotate90 + step + 4) % 4;
    applyRotation({ resetCrop: true });
  }

  function setAngle(value) {
    angle = Math.max(-10, Math.min(10, round1(value)));
    applyRotation({ resetCrop: false });
  }

  function setRatio(name) {
    ratioButtons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.ratio === name)));
    if (name === "free") {
      cropper.setAspectRatio(NaN);
    } else {
      const [w, h] = name.split(":").map(Number);
      cropper.setAspectRatio(w / h);
    }
  }

  function setBusy(value) {
    busy = value;
    document.body.classList.toggle("busy", value);
    document.querySelectorAll(".tools button").forEach((b) => (b.disabled = value));
  }

  async function save(noCrop) {
    if (busy || !cropper) return;
    const d = cropper.getData();
    const body = {
      rotate90,
      angle,
      crop: noCrop
        ? null
        : { x: d.x * scale, y: d.y * scale, w: d.width * scale, h: d.height * scale },
    };
    errorBox.hidden = true;
    setBusy(true);
    try {
      const resp = await fetch(cfg.saveUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.error || `ошибка сервера (${resp.status})`);
      window.location.assign(data.next);
    } catch (e) {
      errorBox.textContent = `Не сохранилось: ${e.message}`;
      errorBox.hidden = false;
      setBusy(false);
    }
  }

  function init() {
    scale = cfg.origWidth / img.naturalWidth;
    const prev = cfg.prev;
    if (prev) {
      rotate90 = ((prev.rotate90 || 0) % 4 + 4) % 4;
      angle = round1(prev.angle || 0);
    }
    cropper = new Cropper(img, {
      viewMode: 1,
      dragMode: "move",
      autoCropArea: 1,
      checkOrientation: false, // превью уже с применённой EXIF-ориентацией
      toggleDragModeOnDblclick: false,
      scalable: false,
      ready() {
        cropper.rotateTo(totalRotation());
        if (prev && prev.crop) {
          cropper.setData({
            x: prev.crop.x / scale,
            y: prev.crop.y / scale,
            width: prev.crop.w / scale,
            height: prev.crop.h / scale,
          });
        } else {
          cropFull();
        }
        showAngle();
        showResult();
      },
      crop() {
        if (cropper) showResult();
      },
    });
  }

  document.querySelector(".tools").addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn || !cropper) return;
    btn.blur(); // иначе Enter после клика ещё раз «нажмёт» эту кнопку
    if (btn.dataset.ratio) return setRatio(btn.dataset.ratio);
    switch (btn.dataset.act) {
      case "rot-left": return rotateQuarter(-1);
      case "rot-right": return rotateQuarter(1);
      case "angle-dec": return setAngle(angle - 0.1);
      case "angle-inc": return setAngle(angle + 0.1);
      case "angle-zero": return setAngle(0);
      case "reset-crop": return cropFull();
      case "save": return save(false);
      case "nocrop": return save(true);
    }
  });

  angleInput.addEventListener("input", () => cropper && setAngle(Number(angleInput.value)));

  document.addEventListener("keydown", (e) => {
    if (busy || !cropper || e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target;
    const typing = t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type !== "range");
    if (typing || t.tagName === "A") return;
    // e.code не зависит от раскладки: R/S работают и в русской
    if (e.code === "KeyR") {
      e.preventDefault();
      rotateQuarter(e.shiftKey ? -1 : 1);
    } else if (e.key === "Enter") {
      e.preventDefault();
      save(false);
    } else if (e.code === "KeyS") {
      e.preventDefault();
      save(true);
    } else if (e.key === "ArrowRight" && t.type !== "range" && cfg.skipUrl) {
      e.preventDefault();
      window.location.assign(cfg.skipUrl);
    }
  });

  if (img.complete && img.naturalWidth) init();
  else img.addEventListener("load", init, { once: true });
})();
