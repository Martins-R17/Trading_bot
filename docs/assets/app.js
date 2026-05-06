(() => {
  const numericValues = document.querySelectorAll("[data-count]");

  for (const node of numericValues) {
    const raw = node.getAttribute("data-count");
    if (!raw) continue;
    const target = Number(raw);
    if (!Number.isFinite(target) || Math.abs(target) > 1_000_000_000) continue;

    const original = node.textContent || "";
    const isMoney = original.trim().startsWith("$");
    const hasDecimal = original.includes(".");
    const start = performance.now();
    const duration = 650;

    const render = (time) => {
      const progress = Math.min((time - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = target * eased;
      const formatted = hasDecimal
        ? current.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
        : Math.round(current).toLocaleString();
      node.textContent = isMoney ? `$${formatted}` : formatted;
      if (progress < 1) requestAnimationFrame(render);
      else node.textContent = original;
    };

    requestAnimationFrame(render);
  }

  const progressBars = document.querySelectorAll("[data-progress] span");
  for (const bar of progressBars) {
    const parent = bar.parentElement;
    const raw = parent ? parent.getAttribute("data-progress") : "";
    const target = Number(raw);
    if (!Number.isFinite(target)) continue;
    bar.style.transform = "scaleX(0)";
    requestAnimationFrame(() => {
      bar.style.transition = "transform 700ms cubic-bezier(.2,.8,.2,1)";
      bar.style.transform = `scaleX(${Math.max(0, Math.min(target, 100)) / 100})`;
    });
  }
})();
