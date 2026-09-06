local abstract_blocks = {}
local in_abstract = false
local skipping_title = true

local function plain_text(block)
  return pandoc.utils.stringify(block)
end

local function normalize_heading(block)
  local heading = pandoc.utils.stringify(block)
  heading = heading:gsub('^%d+%.%d+%.%s*', '')
  heading = heading:gsub('^%d+%.%d+%s*', '')
  heading = heading:gsub('^%d+%.%s*', '')
  block.content = pandoc.List{pandoc.Str(heading)}
  return block
end

function Pandoc(doc)
  local out = {}
  for _, block in ipairs(doc.blocks) do
    if block.t == 'Header' and skipping_title and block.level == 1 then
      skipping_title = false
    elseif block.t == 'Header' and plain_text(block) == 'Abstract' then
      in_abstract = true
    elseif in_abstract and block.t == 'Header' then
      in_abstract = false
      block.level = math.max(1, block.level - 1)
      block = normalize_heading(block)
      table.insert(out, block)
    elseif in_abstract then
      table.insert(abstract_blocks, block)
    else
      if block.t == 'Header' then
        block.level = math.max(1, block.level - 1)
        block = normalize_heading(block)
      end
      table.insert(out, block)
    end
  end
  doc.blocks = out
  doc.meta.title = pandoc.MetaInlines{pandoc.Str('Migration-Budgeted Local Reconfiguration Under Persistent Hotspots in Mobile Edge Systems')}
  doc.meta.abstract = pandoc.MetaBlocks(abstract_blocks)
  doc.meta.keywords = pandoc.MetaList{
    pandoc.MetaString('mobile edge computing'),
    pandoc.MetaString('service migration'),
    pandoc.MetaString('large neighborhood search'),
    pandoc.MetaString('online reconfiguration'),
    pandoc.MetaString('persistent hotspot')
  }
  return doc
end

function Image(img)
  local name = img.src:match('([^/\\]+)%.png$')
  if name and name:match('_PMC$') then img.src = 'figures/' .. name .. '.pdf' end
  return img
end
