// @ts-ignore
import copyMarkdownScript from "./scripts/copymarkdown.inline"
import styles from "./styles/copymarkdown.scss"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"

const CopyMarkdown: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
  return (
    <button
      class={classNames(displayClass, "copy-markdown")}
      aria-label="Copy article as markdown"
      type="button"
    >
      <svg
        class="copy-icon copy-icon-copy"
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
        <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
      </svg>
      <svg
        class="copy-icon copy-icon-check"
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <polyline points="20 6 9 17 4 12" />
      </svg>
      <span class="copy-label copy-label-default">Copy markdown</span>
      <span class="copy-label copy-label-success">Copied</span>
    </button>
  )
}

CopyMarkdown.afterDOMLoaded = copyMarkdownScript
CopyMarkdown.css = styles

export default (() => CopyMarkdown) satisfies QuartzComponentConstructor
