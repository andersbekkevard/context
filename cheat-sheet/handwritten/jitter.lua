-- jitter.lua
-- Natural-noise filter for LuaLaTeX.
-- Adds per-glyph x/y offset and per-line vertical drift so the rendered text
-- breaks the "every glyph identical / baseline laser-straight" tell of TeX.

local glyph_id = node.id("glyph")
local hlist_id = node.id("hlist")
local vlist_id = node.id("vlist")
local PT       = 65536  -- 1pt expressed in scaled points

-- Tunables (in points). Smaller = subtler, larger = wobblier.
local GLYPH_Y    = 0.25   -- ±0.25pt per glyph vertical
local GLYPH_X    = 0.15   -- ±0.15pt per glyph horizontal
local LINE_DRIFT = 0.50   -- ±0.5pt per line vertical shift

math.randomseed(20260518)  -- deterministic seed: exam date

local function jitter_glyph(g)
  g.yoffset = (g.yoffset or 0) + (math.random() - 0.5) * 2 * GLYPH_Y * PT
  g.xoffset = (g.xoffset or 0) + (math.random() - 0.5) * 2 * GLYPH_X * PT
end

-- Walk the node list recursively, jittering every glyph regardless of nesting.
local function walk_glyphs(head)
  local n = head
  while n do
    local id = n.id
    if id == glyph_id then
      jitter_glyph(n)
    elseif id == hlist_id or id == vlist_id then
      walk_glyphs(n.list)
    end
    n = n.next
  end
end

-- For each top-level hlist (i.e. each line of a paragraph after line breaking),
-- nudge the whole line up/down by a small random amount.
local function drift_lines(head)
  local n = head
  while n do
    if n.id == hlist_id then
      n.shift = (n.shift or 0) + (math.random() - 0.5) * 2 * LINE_DRIFT * PT
    end
    n = n.next
  end
end

luatexbase.add_to_callback(
  "post_linebreak_filter",
  function(head, _)
    walk_glyphs(head)
    drift_lines(head)
    return true
  end,
  "handwritten-jitter"
)

-- Also catch glyphs inside hboxes (e.g. table cells, math boxes) that don't
-- pass through post_linebreak_filter.  hpack_filter runs on every \hbox.
luatexbase.add_to_callback(
  "hpack_filter",
  function(head, _)
    walk_glyphs(head)
    return true
  end,
  "handwritten-jitter-hpack"
)
