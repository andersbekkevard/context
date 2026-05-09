import { FilePath, joinSegments } from "../quartz/util/path"
import { QuartzEmitterPlugin } from "../quartz/plugins/types"
import { glob } from "../quartz/util/glob"
import fs from "fs"
import path from "path"

/**
 * Copies the project-level `web/static/` folder verbatim to the build output root.
 *
 * Quartz's built-in Static plugin only copies its engine-internal `quartz/static/`
 * (icons, og-image). User-provided pass-through assets (our deck HTML/CSS/JS at
 * `web/static/decks/...`) need this companion emitter so they land at
 * `public/decks/...` and are served at `/decks/...`.
 */
export const WebStatic: QuartzEmitterPlugin = () => ({
  name: "WebStatic",
  async *emit({ argv }) {
    const staticRoot = path.resolve(process.cwd(), "static")
    if (!fs.existsSync(staticRoot)) return
    const fps = await glob("**", staticRoot, [])
    for (const fp of fps) {
      const src = joinSegments(staticRoot, fp) as FilePath
      const dest = joinSegments(argv.output, fp) as FilePath
      await fs.promises.mkdir(path.dirname(dest), { recursive: true })
      await fs.promises.copyFile(src, dest)
      yield dest
    }
  },
  async *partialEmit(ctx, _content, _resources, changeEvents) {
    const staticRoot = path.resolve(process.cwd(), "static")
    for (const ev of changeEvents) {
      const abs = path.resolve(ev.path)
      if (!abs.startsWith(staticRoot)) continue
      const rel = path.relative(staticRoot, abs)
      const dest = joinSegments(ctx.argv.output, rel) as FilePath
      if (ev.type === "add" || ev.type === "change") {
        await fs.promises.mkdir(path.dirname(dest), { recursive: true })
        await fs.promises.copyFile(abs, dest)
        yield dest
      } else if (ev.type === "delete") {
        try {
          await fs.promises.unlink(dest)
        } catch {}
      }
    }
  },
})
