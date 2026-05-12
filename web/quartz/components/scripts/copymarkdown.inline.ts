document.addEventListener("nav", () => {
  for (const btn of document.getElementsByClassName("copy-markdown")) {
    const button = btn as HTMLButtonElement
    const onClick = async () => {
      try {
        const url =
          location.pathname.replace(/\.html$/, "").replace(/\/$/, "") + ".md"
        const res = await fetch(url)
        if (!res.ok) throw new Error(`fetch ${url} -> ${res.status}`)
        const md = await res.text()
        await navigator.clipboard.writeText(md)
        button.setAttribute("data-copied", "true")
        window.setTimeout(() => button.removeAttribute("data-copied"), 1500)
      } catch (err) {
        console.error("CopyMarkdown failed:", err)
      }
    }
    button.addEventListener("click", onClick)
    window.addCleanup(() => button.removeEventListener("click", onClick))
  }
})
