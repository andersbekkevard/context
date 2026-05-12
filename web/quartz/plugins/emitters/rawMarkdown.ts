import fs from "fs"
import { QuartzEmitterPlugin } from "../types"
import { write } from "./helpers"

export const RawMarkdown: QuartzEmitterPlugin = () => {
  return {
    name: "RawMarkdown",
    getQuartzComponents() {
      return []
    },
    async *emit(ctx, content, _resources) {
      for (const [_tree, file] of content) {
        const slug = file.data.slug!
        const filePath = file.data.filePath
        if (!filePath) continue
        if (slug.endsWith("/index") || slug.startsWith("tags/")) continue

        const source = await fs.promises.readFile(filePath, "utf-8")
        yield write({
          ctx,
          content: source,
          slug,
          ext: ".md",
        })
      }
    },
    async *partialEmit(ctx, content, _resources, changeEvents) {
      const changedSlugs = new Set<string>()
      for (const changeEvent of changeEvents) {
        if (!changeEvent.file) continue
        if (changeEvent.type === "add" || changeEvent.type === "change") {
          changedSlugs.add(changeEvent.file.data.slug!)
        }
      }

      for (const [_tree, file] of content) {
        const slug = file.data.slug!
        if (!changedSlugs.has(slug)) continue
        const filePath = file.data.filePath
        if (!filePath) continue
        if (slug.endsWith("/index") || slug.startsWith("tags/")) continue

        const source = await fs.promises.readFile(filePath, "utf-8")
        yield write({
          ctx,
          content: source,
          slug,
          ext: ".md",
        })
      }
    },
  }
}
