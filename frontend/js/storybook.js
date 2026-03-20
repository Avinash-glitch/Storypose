/**
 * storybook.js — Page rendering and storybook layout.
 *
 * Exposes:
 *   Storybook.addPage(imageBase64, storyText, poseDesc, pageNum) → void
 *   Storybook.renderCover(title, pageCount)                       → void
 *   Storybook.showWelcome()                                       → void
 *   Storybook.hideWelcome()                                       → void
 *   Storybook.downloadStory()                                     → void
 */

const Storybook = (() => {
  const container    = document.getElementById("storybook-container");
  const welcomeEl    = document.getElementById("welcome-screen");
  const pageCountEl  = document.getElementById("page-count");

  let pageCount = 0;

  // -----------------------------------------------------------------------
  // Internal helpers
  // -----------------------------------------------------------------------

  function scrollToBottom() {
    setTimeout(() => {
      container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
    }, 100);
  }

  function incrementPageCount() {
    pageCount++;
    if (pageCountEl) pageCountEl.textContent = pageCount;
  }

  function animatePage(el) {
    el.classList.add("page-turn-anim");
    el.addEventListener("animationend", () => el.classList.remove("page-turn-anim"), {
      once: true,
    });
  }

  // -----------------------------------------------------------------------
  // addPage — render a storybook spread (illustration + text)
  // -----------------------------------------------------------------------
  function addPage(imageBase64, storyText, poseDesc, pageNum) {
    const spread = document.createElement("div");
    spread.className = "storybook-page";

    // --- Left page: text ---
    const leftPage = document.createElement("div");
    leftPage.className = "page page-left";

    const textEl = document.createElement("p");
    textEl.className = "story-text";
    textEl.textContent = storyText;
    leftPage.appendChild(textEl);

    if (poseDesc && poseDesc !== "standing naturally") {
      const poseEl = document.createElement("p");
      poseEl.className = "pose-tag";
      poseEl.textContent = `Pose: ${poseDesc}`;
      leftPage.appendChild(poseEl);
    }

    const leftNum = document.createElement("span");
    leftNum.className = "page-number page-number-left";
    leftNum.textContent = pageNum * 2 - 1;
    leftPage.appendChild(leftNum);

    // --- Right page: illustration ---
    const rightPage = document.createElement("div");
    rightPage.className = "page page-right";

    if (imageBase64) {
      const img = document.createElement("img");
      img.className = "story-illustration";
      img.src = `data:image/png;base64,${imageBase64}`;
      img.alt = `Storybook illustration — page ${pageNum}`;
      rightPage.appendChild(img);
    } else {
      const placeholder = document.createElement("div");
      placeholder.className = "story-illustration-placeholder";
      placeholder.textContent = "🎨";
      rightPage.appendChild(placeholder);
    }

    const rightNum = document.createElement("span");
    rightNum.className = "page-number page-number-right";
    rightNum.textContent = pageNum * 2;
    rightPage.appendChild(rightNum);

    spread.appendChild(leftPage);
    spread.appendChild(rightPage);
    container.appendChild(spread);

    animatePage(spread);
    incrementPageCount();
    scrollToBottom();
  }

  // -----------------------------------------------------------------------
  // renderCover — decorative cover page at the top
  // -----------------------------------------------------------------------
  function renderCover(title, totalPages) {
    const cover = document.createElement("div");
    cover.className = "storybook-cover";

    const icon = document.createElement("div");
    icon.style.fontSize = "4rem";
    icon.textContent = "📖";
    cover.appendChild(icon);

    const titleEl = document.createElement("h1");
    titleEl.className = "cover-title";
    titleEl.textContent = title || "My Amazing Story";
    cover.appendChild(titleEl);

    const sub = document.createElement("p");
    sub.className = "cover-subtitle";
    sub.textContent = `${totalPages} page${totalPages !== 1 ? "s" : ""} of adventure`;
    cover.appendChild(sub);

    // Prepend to the top of the storybook
    container.insertBefore(cover, container.firstChild);
    container.scrollTo({ top: 0, behavior: "smooth" });
  }

  // -----------------------------------------------------------------------
  // Visibility helpers
  // -----------------------------------------------------------------------
  function showWelcome() {
    if (welcomeEl) welcomeEl.style.display = "flex";
    container.style.display = "none";
  }

  function hideWelcome() {
    if (welcomeEl) welcomeEl.style.display = "none";
    container.style.display = "flex";
  }

  // -----------------------------------------------------------------------
  // downloadStory — save the full storybook as a printable HTML file
  // -----------------------------------------------------------------------
  function downloadStory() {
    const pages = [];
    container.querySelectorAll(".storybook-page").forEach((spread) => {
      const img   = spread.querySelector("img");
      const text  = spread.querySelector(".story-text");
      const poseTag = spread.querySelector(".pose-tag");
      pages.push({
        src:  img  ? img.src  : null,
        text: text ? text.textContent : "",
        pose: poseTag ? poseTag.textContent : "",
      });
    });

    const cover = container.querySelector(".storybook-cover");
    const coverTitle = cover ? cover.querySelector(".cover-title")?.textContent : "My Story";

    const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>${coverTitle}</title>
  <style>
    body { font-family: 'Patrick Hand', cursive; background: #fdf6e3; margin: 0; padding: 20px; }
    .page-spread { display: flex; margin-bottom: 40px; border: 2px solid #8B7355; border-radius: 6px; overflow: hidden; page-break-after: always; }
    .page { flex: 1; padding: 24px; background: #fdf6e3; }
    .page-left { border-right: 2px solid #8B7355; display: flex; align-items: center; justify-content: center; }
    img { max-width: 100%; border-radius: 8px; }
    p { font-size: 1.1rem; line-height: 1.9; color: #3d2b1f; }
    h1 { font-size: 2rem; color: #e07b39; text-align: center; margin-bottom: 40px; }
    .pose-tag { font-size: 0.75rem; color: #999; font-style: italic; }
    @media print { .page-spread { page-break-after: always; } }
  </style>
</head>
<body>
  <h1>📖 ${coverTitle}</h1>
  ${pages.map((p, i) => `
  <div class="page-spread">
    <div class="page page-left">
      ${p.src ? `<img src="${p.src}" alt="Page ${i + 1}" />` : "<p>🎨</p>"}
    </div>
    <div class="page">
      <p>${p.text}</p>
      ${p.pose ? `<p class="pose-tag">${p.pose}</p>` : ""}
    </div>
  </div>`).join("")}
</body>
</html>`;

    const blob = new Blob([html], { type: "text/html" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `${(coverTitle || "my-story").replace(/\s+/g, "-").toLowerCase()}.html`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function reset() {
    // Remove all page spreads (keep welcome screen)
    Array.from(container.children).forEach(child => {
      if (!child.id || child.id !== "welcome-screen") child.remove();
    });
    pageCount = 0;
    if (pageCountEl) pageCountEl.textContent = 0;
    showWelcome();
  }

  return { addPage, renderCover, showWelcome, hideWelcome, downloadStory, reset };
})();
