document.addEventListener("DOMContentLoaded", () => {
  "use strict";

  const currentPath = window.location.pathname || "/";
  const siteMarker = "/_site/";
  const filePrefix = window.location.protocol === "file:" && currentPath.includes(siteMarker)
    ? currentPath.slice(0, currentPath.indexOf(siteMarker) + siteMarker.length - 1)
    : "";
  const sitePath = filePrefix ? currentPath.slice(filePrefix.length) || "/" : currentPath;
  const isChinese = /^\/zh(?:\/|$)/.test(sitePath);

  const translatedPath = (targetLanguage) => {
    let path = sitePath;
    if (targetLanguage === "zh") {
      if (!/^\/zh(?:\/|$)/.test(path)) {
        path = path === "/" ? "/zh/" : `/zh${path}`;
      }
    } else {
      path = path.replace(/^\/zh(?=\/|$)/, "") || "/";
    }
    return `${filePrefix}${path}${window.location.search}${window.location.hash}`;
  };

  const navAnchor = (id, label) => {
    const node = document.getElementById(id);
    if (node) return node.matches("a") ? node : node.querySelector("a");
    return [...document.querySelectorAll(".navbar-nav.ms-auto .nav-link")]
      .find((link) => link.textContent.trim() === label) || null;
  };

  const zhLink = navAnchor("lang-zh", "中文");
  const enLink = navAnchor("lang-en", "English");

  if (zhLink) {
    zhLink.id = "lang-zh";
    zhLink.href = translatedPath("zh");
    zhLink.classList.toggle("lang-active", isChinese);
    if (isChinese) zhLink.setAttribute("aria-current", "page");
    else zhLink.removeAttribute("aria-current");
  }

  if (enLink) {
    enLink.id = "lang-en";
    enLink.href = translatedPath("en");
    enLink.classList.toggle("lang-active", !isChinese);
    if (!isChinese) enLink.setAttribute("aria-current", "page");
    else enLink.removeAttribute("aria-current");
  }

  const navLabels = ["首页", "博客"];
  const primaryLinks = [...document.querySelectorAll(".navbar .navbar-nav.me-auto .nav-link")];
  const sectionIndex = /\/blog(?:\.html|\/|$)/.test(sitePath) ? 1 : 0;

  primaryLinks.slice(0, navLabels.length).forEach((link, index) => {
    link.classList.toggle("active", index === sectionIndex);
    if (index === sectionIndex) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });

  const isBlog = /\/blog(?:\.html|\/|$)/.test(sitePath);
  const decorateBlogListing = () => {
    if (!isBlog) return;

    const categoryTitle = document.querySelector(".quarto-listing-category-title");
    const categoryTree = document.querySelector(".quarto-listing-category");
    const categoryDefinitions = isChinese
      ? [
          { label: "研究", branch: "research" },
          { label: "思考", branch: "reflection" },
          { label: "观察", branch: "observation" }
        ]
      : [
          { label: "Research", branch: "research" },
          { label: "Thinking", branch: "reflection" },
          { label: "Observation", branch: "observation" }
        ];
    const categoryToken = (label) => btoa(encodeURIComponent(label));
    const categoryLabel = (node) => {
      const textNode = [...node.childNodes]
        .find((child) => child.nodeType === Node.TEXT_NODE);
      return (textNode?.textContent || node.textContent)
        .replace(/\s*\(\d+\)\s*$/, "")
        .trim();
    };

    if (categoryTitle) categoryTitle.textContent = isChinese ? "内容地图" : "Content map";
    if (categoryTree) {
      categoryTree.classList.add("category-tree");
      categoryTree.setAttribute("aria-label", isChinese ? "文章分类树" : "Post category tree");

      categoryDefinitions.forEach(({ label }) => {
        const exists = [...categoryTree.querySelectorAll(".category")]
          .some((node) => categoryLabel(node) === label);
        if (exists) return;

        const node = document.createElement("div");
        const count = document.createElement("span");
        node.className = "category";
        node.dataset.category = categoryToken(label);
        node.append(document.createTextNode(`${label} `));
        count.className = "quarto-category-count";
        count.textContent = "(0)";
        node.append(count);
        node.addEventListener("click", () => {
          window.quartoListingCategory?.(node.dataset.category);
        });
        categoryTree.append(node);
      });
    }

    const categories = [...document.querySelectorAll(".quarto-listing-category .category")];
    categories.forEach((node) => {
      const textNode = [...node.childNodes].find((child) => child.nodeType === Node.TEXT_NODE);
      const label = categoryLabel(node);
      const isRoot = node.dataset.category === "" || /^(All|全部|所有)$/.test(label);
      const isResearch = /^(Research|研究)$/.test(label);
      const isReflection = /^(Thinking|思考)$/.test(label);
      const isObservation = /^(Observation|观察)$/.test(label);

      node.classList.toggle("tree-root", isRoot);
      node.classList.toggle("tree-branch", !isRoot);
      node.classList.toggle("branch-research", isResearch);
      node.classList.toggle("branch-reflection", isReflection);
      node.classList.toggle("branch-observation", isObservation);
      node.setAttribute("role", "button");
      node.setAttribute("tabindex", "0");
      node.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          node.click();
        }
      });

      if (isRoot && textNode) textNode.textContent = isChinese ? "所有 " : "All ";
    });

    if (categoryTree) {
      const orderedCategories = [
        categories.find((node) => node.classList.contains("tree-root")),
        categories.find((node) => node.classList.contains("branch-research")),
        categories.find((node) => node.classList.contains("branch-reflection")),
        categories.find((node) => node.classList.contains("branch-observation"))
      ].filter(Boolean);
      orderedCategories.forEach((node) => categoryTree.appendChild(node));
    }

    if (!categories.some((node) => node.classList.contains("active"))) {
      const activeLabel = new URLSearchParams(window.location.hash.slice(1)).get("category");
      const activeNode = activeLabel
        ? categories.find((node) => categoryLabel(node) === activeLabel)
        : categories.find((node) => node.classList.contains("tree-root"));
      activeNode?.classList.add("active");
    }

    document.querySelectorAll(".quarto-post").forEach((post) => {
      const labels = [...post.querySelectorAll(".listing-category")]
        .map((node) => node.textContent.trim());
      const research = labels.some((label) => /^(Research|研究)$/.test(label));
      const reflection = labels.some((label) => /^(Thinking|思考)$/.test(label));
      const observation = labels.some((label) => /^(Observation|观察)$/.test(label));
      post.classList.toggle("post-research", research);
      post.classList.toggle("post-reflection", reflection);
      post.classList.toggle("post-observation", observation);
      if (research) post.dataset.section = "research";
      if (reflection) post.dataset.section = "reflection";
      if (observation) post.dataset.section = "observation";
    });
  };

  const noteNumber = (value) => value.match(/\d+/)?.[0] || null;
  const marginNotePairs = () => {
    const markers = new Map();
    document.querySelectorAll(".margin-note-ref").forEach((marker) => {
      const number = noteNumber(marker.getAttribute("aria-label") || marker.textContent);
      if (number && !markers.has(number)) markers.set(number, marker);
    });

    return [...document.querySelectorAll(".column-margin .margin-label")]
      .map((label) => {
        const number = noteNumber(label.textContent);
        const container = label.closest(".column-margin");
        let note = label;
        while (note.parentElement && note.parentElement !== container) {
          note = note.parentElement;
        }
        return { marker: markers.get(number), label, note };
      })
      .filter(({ marker, note }) => marker && note);
  };

  const alignMarginNotes = () => {
    const pairs = marginNotePairs();
    pairs.forEach(({ note }) => {
      note.classList.add("js-margin-note-aligned");
      note.style.setProperty("--margin-note-shift", "0px");
    });

    if (!window.matchMedia("(min-width: 992px)").matches) {
      pairs.forEach(({ note }) => note.style.removeProperty("--margin-note-shift"));
      return;
    }

    window.requestAnimationFrame(() => {
      const gap = 16;
      let previousBottom = Number.NEGATIVE_INFINITY;

      pairs
        .map((pair) => ({
          ...pair,
          markerTop: pair.marker.getBoundingClientRect().top
        }))
        .sort((a, b) => a.markerTop - b.markerTop)
        .forEach(({ markerTop, label, note }) => {
          const labelBox = label.getBoundingClientRect();
          const noteBox = note.getBoundingClientRect();
          let shift = markerTop - labelBox.top;
          const noteTop = noteBox.top + shift;

          if (noteTop < previousBottom + gap) {
            shift += previousBottom + gap - noteTop;
          }

          note.style.setProperty("--margin-note-shift", `${Math.round(shift)}px`);
          previousBottom = noteBox.bottom + shift;
        });
    });
  };

  if (document.querySelector(".margin-note-ref")) {
    let alignmentTimer;
    const scheduleMarginNoteAlignment = () => {
      window.clearTimeout(alignmentTimer);
      alignmentTimer = window.setTimeout(alignMarginNotes, 60);
    };

    window.addEventListener("load", scheduleMarginNoteAlignment);
    window.addEventListener("resize", scheduleMarginNoteAlignment);
    document.fonts?.ready.then(scheduleMarginNoteAlignment);
    document.querySelectorAll("#quarto-document-content img").forEach((image) => {
      if (!image.complete) {
        image.addEventListener("load", scheduleMarginNoteAlignment, { once: true });
      }
    });

    const article = document.querySelector("#quarto-document-content");
    if (article && "ResizeObserver" in window) {
      new ResizeObserver(scheduleMarginNoteAlignment).observe(article);
    }

    scheduleMarginNoteAlignment();
  }

  if (!isChinese) {
    decorateBlogListing();
    return;
  }

  document.documentElement.lang = "zh-CN";

  const chineseNavPaths = ["/zh/", "/zh/blog.html"];
  primaryLinks.slice(0, navLabels.length).forEach((link, index) => {
    const label = link.querySelector(".menu-text") || link;
    if (navLabels[index]) label.textContent = navLabels[index];
    if (chineseNavPaths[index]) link.href = `${filePrefix}${chineseNavPaths[index]}`;
  });

  const brand = document.querySelector(".navbar-brand");
  if (brand) brand.href = `${filePrefix}/zh/`;

  const footerLeft = document.querySelector(".nav-footer-left");
  const footerRight = document.querySelector(".nav-footer-right");
  if (footerLeft) footerLeft.textContent = "© 2026 Yingjie Tan";
  if (footerRight) {
    const contactLink = document.createElement("a");
    contactLink.href = "mailto:tanyj23@mails.tsinghua.edu.cn";
    contactLink.textContent = "联系我";
    contactLink.setAttribute("aria-label", "发送邮件至 tanyj23@mails.tsinghua.edu.cn");
    footerRight.replaceChildren(contactLink);
  }

  document.querySelectorAll(".quarto-listing-category-title").forEach((node) => {
    node.textContent = isBlog ? "内容地图" : "分类";
  });

  document.querySelectorAll(".quarto-listing-category .category").forEach((node) => {
    if (node.firstChild?.nodeType === Node.TEXT_NODE) {
      node.firstChild.textContent = node.firstChild.textContent.replace(/^(All|全部)/, "所有");
    }
  });

  document.querySelectorAll(".quarto-listing-filter input").forEach((input) => {
    input.placeholder = "筛选文章";
    input.setAttribute("aria-label", "筛选文章");
  });

  document.querySelectorAll(".quarto-post .metadata").forEach((node) => {
    const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      walker.currentNode.textContent = walker.currentNode.textContent
        .replace(/\bmins?\b/g, "分钟")
        .replace(/\bmin\b/g, "分钟");
    }
  });

  const titleMeta = {
    Author: "作者",
    Published: "发布日期",
    Modified: "更新日期"
  };
  document.querySelectorAll(".quarto-title-meta-heading").forEach((node) => {
    const source = node.textContent.trim();
    if (titleMeta[source]) node.textContent = titleMeta[source];
  });

  document.querySelectorAll(".code-copy-button").forEach((button) => {
    button.title = "复制代码";
    button.setAttribute("aria-label", "复制代码");
  });

  const backToTop = document.querySelector(".back-to-top-navigation");
  if (backToTop) {
    backToTop.setAttribute("aria-label", "返回顶部");
    backToTop.title = "返回顶部";
  }

  decorateBlogListing();
});
