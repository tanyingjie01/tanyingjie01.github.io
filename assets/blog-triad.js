(() => {
  "use strict";

  document.querySelectorAll(".blog-triad").forEach((diagram) => {
    const panels = [...diagram.querySelectorAll(".blog-triad__panel")];
    const controls = [...diagram.querySelectorAll(".blog-triad__control")];
    let hovered = null;
    let focused = null;
    let selected = null;

    const render = () => {
      const active = hovered || focused || selected;
      if (active) diagram.dataset.active = active;
      else delete diagram.dataset.active;

      panels.forEach((panel) => {
        const visible = panel.dataset.panel === (active || "intro");
        panel.classList.toggle("is-visible", visible);
        panel.setAttribute("aria-hidden", String(!visible));
      });
      controls.forEach((control) => {
        control.classList.toggle("is-active", control.dataset.part === active);
        control.setAttribute("aria-pressed", String(control.dataset.part === selected));
      });
    };

    // A filled hit region includes each symbol's negative space, so the
    // illustration does not flicker while the pointer crosses its fine lines.
    diagram.querySelectorAll("[data-interactive]").forEach((target) => {
      target.addEventListener("pointerenter", (event) => {
        if (event.pointerType === "touch") return;
        hovered = target.dataset.part;
        render();
      });
      target.addEventListener("pointerleave", (event) => {
        if (event.pointerType === "touch") return;
        hovered = null;
        render();
      });
      target.addEventListener("click", () => {
        selected = selected === target.dataset.part ? null : target.dataset.part;
        // A second click really returns to the introduction, even while the
        // pointer or keyboard focus remains on the same control.
        hovered = focused = null;
        render();
      });
    });

    controls.forEach((control) => {
      control.addEventListener("focus", () => {
        focused = control.dataset.part;
        render();
      });
      control.addEventListener("blur", () => {
        focused = null;
        render();
      });
    });

    diagram.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      hovered = focused = selected = null;
      render();
    });

    render();
  });
})();
