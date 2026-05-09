import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/footer.scss"

interface Options {
  links: Record<string, string>
}

export default ((_opts?: Options) => {
  const Footer: QuartzComponent = (_props: QuartzComponentProps) => null
  Footer.css = style
  return Footer
}) satisfies QuartzComponentConstructor
