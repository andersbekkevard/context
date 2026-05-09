import { FilePath, joinSegments } from "../quartz/util/path"
import { QuartzEmitterPlugin } from "../quartz/plugins/types"
import { glob } from "../quartz/util/glob"
import fs from "fs"
import path from "path"

/**
 * Copies the repo-root `pdfs/` folder verbatim into `public/pdfs/`.
 *
 * The pdfs/ folder holds curated, standardized-filename copies of the course
 * slide decks, recommended exercises, and compulsory exercises. They live
 * outside `wiki/` (so LLMs don't try to load binary PDFs as context) and
 * outside `web/` (so the deletable web layer rule still holds), as a
 * sibling folder. This emitter pulls them into the build output so Vercel
 * serves them at /pdfs/<filename>.
 *
 * Mirror of WebStatic but with a different source root and namespaced
 * destination (`public/pdfs/`). Kept separate so each emitter has one job.
 */
export const CoursePdfs: QuartzEmitterPlugin = () => ({
  name: "CoursePdfs",
  async *emit({ argv }) {
    const sourceRoot = path.resolve(process.cwd(), "..", "pdfs")
    if (!fs.existsSync(sourceRoot)) return
    const fps = await glob("**/*.pdf", sourceRoot, [])
    for (const fp of fps) {
      const src = joinSegments(sourceRoot, fp) as FilePath
      const dest = joinSegments(argv.output, "pdfs", fp) as FilePath
      await fs.promises.mkdir(path.dirname(dest), { recursive: true })
      await fs.promises.copyFile(src, dest)
      yield dest
    }
  },
  async *partialEmit(ctx, _content, _resources, changeEvents) {
    const sourceRoot = path.resolve(process.cwd(), "..", "pdfs")
    for (const ev of changeEvents) {
      const abs = path.resolve(ev.path)
      if (!abs.startsWith(sourceRoot)) continue
      if (!abs.endsWith(".pdf")) continue
      const rel = path.relative(sourceRoot, abs)
      const dest = joinSegments(ctx.argv.output, "pdfs", rel) as FilePath
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
