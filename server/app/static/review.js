// Экран проверки: «два рядом» со слайдером и сетка 2×2. Зум и сдвиг общие для всех
// панелей; при зуме выше превью подгружается полный файл.
(() => {
  const cfg = JSON.parse(document.getElementById("review-config").textContent);
  const images = new Map(cfg.images.map((i) => [i.key, i]));
  const variants = cfg.images.filter((i) => i.key !== "prepared");
  const PREVIEW_PX = 2048;
  const MAX_SCALE = 40;

  const view = { s: 1, x: 0, y: 0 }; // s = 1 — вписано в панель
  const state = {
    mode: "compare",
    left: "prepared",
    right: cfg.selected || "prepared",
    selected: cfg.selected,
    split: 0.5,
  };

  const compare = document.getElementById("compare");
  const grid = document.getElementById("grid");
  const zoomOut = document.getElementById("zoom-out");

  // --- клавиши работают и без вариантов ------------------------------------------------
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || document.querySelector("dialog[open]")) return;
    const t = e.target;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName) || t.isContentEditable) return;
    if (e.key === "ArrowLeft" && cfg.prevUrl) return window.location.assign(cfg.prevUrl);
    if (e.key === "ArrowRight" && cfg.nextUrl) return window.location.assign(cfg.nextUrl);
    if (/^[1-9]$/.test(e.key)) {
      const tile = [...document.querySelectorAll(".tile")]
        .find((el) => el.querySelector(".n").textContent.trim() === e.key);
      if (tile) {
        e.preventDefault();
        select(tile.dataset.key);
      }
    }
  });
  if (!compare) return;

  // --- что где показано -----------------------------------------------------------------
  function slots() {
    if (state.mode === "compare") return { left: state.left, right: state.right };
    const sel = images.get(state.selected);
    const round = sel ? sel.round : variants[0]?.round;
    const same = variants.filter((v) => v.round === round && !v.hidden);
    const rest = variants.filter((v) => !same.includes(v) && !v.hidden);
    const picked = [...same, ...rest].slice(0, 3).map((v) => v.key);
    return { g0: "prepared", g1: picked[0], g2: picked[1], g3: picked[2] };
  }

  function stages() {
    return state.mode === "compare" ? [compare] : [...grid.querySelectorAll(".vstage")];
  }

  function render() {
    const s = slots();
    for (const [slot, key] of Object.entries(s)) {
      const layer = document.querySelector(`[data-slot="${slot}"]`);
      const tag = document.querySelector(`[data-tag="${slot}"]`);
      const entry = key && images.get(key);
      layer.hidden = !entry;
      if (tag) tag.textContent = entry ? label(entry) : "";
      if (!entry) continue;
      const img = layer.querySelector("img");
      if (img.dataset.key !== key) {
        img.dataset.key = key;
        img.dataset.full = "";
        img.src = entry.preview;
        img.alt = label(entry);
      }
    }
    compare.querySelector(".over").style.clipPath = `inset(0 ${(1 - state.split) * 100}% 0 0)`;
    compare.querySelector(".divider").style.left = `${state.split * 100}%`;
    apply();
  }

  function label(entry) {
    return entry.key === "prepared" ? entry.label : `${entry.label} · ${entry.badge}`;
  }

  // --- зум и сдвиг ----------------------------------------------------------------------
  function fit(entry, stage) {
    return Math.min(stage.clientWidth / entry.width, stage.clientHeight / entry.height);
  }

  function reference() {
    const key = state.mode === "compare" ? state.right : state.selected || "prepared";
    return images.get(key) || images.get("prepared");
  }

  function apply() {
    const transform = `translate(${view.x}px, ${view.y}px) scale(${view.s})`;
    for (const stage of stages()) {
      for (const img of stage.querySelectorAll("img")) {
        img.style.transform = transform;
        upgrade(img, stage);
      }
    }
    const stage = stages()[0];
    const pct = Math.round(view.s * fit(reference(), stage) * 100);
    zoomOut.textContent = view.s === 1 ? `вписано · ${pct}%` : `${pct}%`;
  }

  // превью — 2048 px по длинной стороне; глубже — полный файл
  function upgrade(img, stage) {
    const entry = images.get(img.dataset.key);
    if (!entry || img.dataset.full) return;
    const previewW = entry.width * Math.min(1, PREVIEW_PX / Math.max(entry.width, entry.height));
    const shownW = entry.width * fit(entry, stage) * view.s * (window.devicePixelRatio || 1);
    if (shownW > previewW * 1.05 && entry.width > previewW) {
      img.dataset.full = "1";
      img.src = entry.full;
    }
  }

  function zoomAt(px, py, scale) {
    const s = Math.min(MAX_SCALE, Math.max(1, scale));
    if (s === 1) {
      Object.assign(view, { s: 1, x: 0, y: 0 });
    } else {
      view.x = px - ((px - view.x) * s) / view.s;
      view.y = py - ((py - view.y) * s) / view.s;
      view.s = s;
      clampPan();
    }
    apply();
  }

  function clampPan() {
    const stage = stages()[0];
    const w = stage.clientWidth;
    const h = stage.clientHeight;
    view.x = Math.min(0, Math.max(w - w * view.s, view.x));
    view.y = Math.min(0, Math.max(h - h * view.s, view.y));
  }

  function oneToOne(px, py) {
    const stage = stages()[0];
    zoomAt(px ?? stage.clientWidth / 2, py ?? stage.clientHeight / 2, 1 / fit(reference(), stage));
  }

  function local(stage, e) {
    const r = stage.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  const pointers = new Map();
  let pinch = null;
  let dragSplit = false;

  function bindStage(stage) {
    stage.addEventListener("wheel", (e) => {
      e.preventDefault();
      const [px, py] = local(stage, e);
      zoomAt(px, py, view.s * Math.exp(-e.deltaY * 0.0015));
    }, { passive: false });

    stage.addEventListener("dblclick", (e) => {
      const [px, py] = local(stage, e);
      if (view.s > 1.01) zoomAt(px, py, 1);
      else oneToOne(px, py);
    });

    stage.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      dragSplit = Boolean(e.target.closest(".divider"));
      stage.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), s: view.s };
      }
      stage.classList.add("grabbing");
    });

    stage.addEventListener("pointermove", (e) => {
      if (!pointers.has(e.pointerId)) return;
      const [lx, ly] = pointers.get(e.pointerId);
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      if (dragSplit && pointers.size === 1) {
        const r = stage.getBoundingClientRect();
        state.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
        return render();
      }
      if (pinch && pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const r = stage.getBoundingClientRect();
        const mx = (a[0] + b[0]) / 2 - r.left;
        const my = (a[1] + b[1]) / 2 - r.top;
        return zoomAt(mx, my, (pinch.s * Math.hypot(a[0] - b[0], a[1] - b[1])) / pinch.d);
      }
      if (view.s > 1) {
        view.x += e.clientX - lx;
        view.y += e.clientY - ly;
        clampPan();
        apply();
      }
    });

    const end = (e) => {
      pointers.delete(e.pointerId);
      if (pointers.size < 2) pinch = null;
      if (!pointers.size) {
        dragSplit = false;
        stage.classList.remove("grabbing");
      }
    };
    stage.addEventListener("pointerup", end);
    stage.addEventListener("pointercancel", end);
  }

  bindStage(compare);
  grid.querySelectorAll(".vstage").forEach(bindStage);
  window.addEventListener("resize", () => {
    if (view.s > 1) clampPan();
    apply();
  });

  // --- панель управления ----------------------------------------------------------------
  const pickLeft = document.getElementById("pick-left");
  const pickRight = document.getElementById("pick-right");
  for (const select of [pickLeft, pickRight]) {
    for (const entry of cfg.images) {
      const opt = document.createElement("option");
      opt.value = entry.key;
      opt.textContent = entry.key === "prepared" ? entry.label : `${entry.label} · ${entry.badge} · раунд ${entry.round}`;
      select.append(opt);
    }
  }
  pickLeft.addEventListener("change", () => {
    state.left = pickLeft.value;
    render();
  });
  pickRight.addEventListener("change", () => {
    state.right = pickRight.value;
    render();
  });

  document.querySelectorAll("[data-mode]").forEach((btn) =>
    btn.addEventListener("click", () => {
      state.mode = btn.dataset.mode;
      document.querySelectorAll("[data-mode]").forEach((b) =>
        b.setAttribute("aria-pressed", String(b === btn)));
      compare.hidden = state.mode !== "compare";
      grid.hidden = state.mode !== "grid";
      document.querySelectorAll(".pair").forEach((el) => (el.hidden = state.mode !== "compare"));
      Object.assign(view, { s: 1, x: 0, y: 0 });
      render();
    }));

  document.querySelector('[data-zoom="fit"]').addEventListener("click", () => zoomAt(0, 0, 1));
  document.querySelector('[data-zoom="100"]').addEventListener("click", () => oneToOne());

  // --- выбор варианта ------------------------------------------------------------------
  const actions = document.getElementById("variant-actions");
  const srcLabel = document.getElementById("src-label");
  const srcVariant = document.getElementById("src-variant");

  function select(key) {
    const entry = images.get(key);
    if (!entry) return;
    state.selected = key;
    state.right = key;
    document.querySelectorAll(".tile").forEach((t) =>
      t.setAttribute("aria-pressed", String(t.dataset.key === key)));
    actions.hidden = false;
    document.getElementById("sel-label").textContent = `${entry.label} · ${entry.badge}`;
    document.getElementById("sel-download").href = entry.download;
    const vis = document.getElementById("sel-visibility");
    vis.action = `/variants/${entry.id}/${entry.hidden ? "show" : "hide"}`;
    document.getElementById("sel-visibility-btn").textContent = entry.hidden ? "Вернуть" : "Скрыть";
    if (srcLabel) {
      srcLabel.textContent = entry.label;
      srcVariant.value = entry.id;
    }
    syncPickers();
    render();
    const url = new URL(window.location);
    url.searchParams.set("v", entry.id);
    history.replaceState(null, "", url);
  }

  function syncPickers() {
    pickLeft.value = state.left;
    pickRight.value = state.right;
  }

  document.querySelectorAll(".tile").forEach((tile) =>
    tile.addEventListener("click", () => {
      tile.blur();
      select(tile.dataset.key);
    }));

  if (state.selected) select(state.selected);
  else render();
})();
