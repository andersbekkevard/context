-- Replace tikzpicture blocks (which pandoc can't render in HTML) with a
-- styled "figure omitted — see PDF" note, so readers know something was
-- there rather than being silently dropped.
function RawBlock(el)
  if el.format == "latex" or el.format == "tex" then
    if el.text:match("\\begin%s*{tikzpicture}") then
      return pandoc.RawBlock(
        "html",
        '<div class="figure-omitted">Figure (TikZ) — rendered in the PDF version only.</div>'
      )
    end
  end
  return nil
end

function RawInline(el)
  if el.format == "latex" or el.format == "tex" then
    if el.text:match("\\begin%s*{tikzpicture}") then
      return pandoc.RawInline(
        "html",
        '<span class="figure-omitted">[TikZ figure — see PDF]</span>'
      )
    end
  end
  return nil
end
