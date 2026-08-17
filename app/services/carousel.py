"""Photo carousels — "the creativity of God in <subject>".

A cover slide with a large title, then real photographs of real places, each
labelled with its location and photographer. Published to Instagram as a
carousel.

The imagery comes from Wikimedia Commons (see wikimedia.py), never from image
generation: this format's whole appeal is that the places are real, and its
audience checks. Locations that cannot be parsed confidently are left off the
slide rather than guessed.

Instagram's Content Publishing API accepts at most 10 items in a carousel, so
sets are capped there even though the app itself allows more.
"""

from __future__ import annotations

import os
import random
import re
import time
from typing import Optional

from loguru import logger
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from app.config import config
from app.services import wikimedia
from app.services import quality, typography as ty
from app.services.verse_card import _cover

# 4:5, the tallest ratio Instagram allows in feed. Rendered at 1440 rather than
# the classic 1080 because Instagram serves feed images up to 1440px on
# high-DPI screens, and giving its transcoder more than it needs is what
# survives re-compression. wikimedia.py sizes and filters every candidate
# against this exact frame, so take it from there rather than restating it.
WIDTH, HEIGHT = wikimedia.TARGET_W, wikimedia.TARGET_H
MAX_SLIDES = 10                      # hard API limit for carousel children

# A slide may never be built by enlarging its photograph. Commons caps
# thumbnails at 3840px, so a source can clear the search filter and still
# arrive too small — the tolerance is for rounding, not for a real upscale.
MAX_UPSCALE = 1.02
# Type roles come from typography.py; see that module for why these faces.

OUT_DIR = "/influencer-automation-2.0/storage/carousels"

# Subject -> (title noun, Commons search terms). Keeping the search terms
# explicit avoids drifting into categories with poor or irrelevant imagery.
# subject -> (title noun, Commons query, extra pool, label_locations)
#
# label_locations is False for wildlife, space and macro detail: those titles
# carry species names and camera metadata ("glaucidium sij", "magnif. ratio")
# which parse as places and would print as a location. A landscape's title
# usually does name a real place; a galaxy has no place at all.
SUBJECTS = {
    # --- landscape ---
    "mountains":    ("MOUNTAINS",       "mountain landscape peak", None, True),
    "auroras":      ("THE AURORAS",     "aurora borealis", None, True),
    "sunsets":      ("THE SUNSET",      "sunset clouds sky", None, True),
    "oceans":       ("THE OCEAN",       "ocean coast sea waves", None, True),
    "forests":      ("THE FORESTS",     "(forest OR woodland OR forêt)", None, True),
    "deserts":      ("THE DESERT",      "(desert OR dune OR dunes)", None, True),
    "rivers":       ("THE RIVERS",      "river valley", None, True),
    "storms":       ("THE STORM",       "(storm OR lightning OR thunderstorm)", None, True),
    "glaciers":     ("THE ICE",         "(glacier OR iceberg OR glaciar)", None, True),
    # No bare "milky": it matched an iceberg whose description mentions the
    # Milky Way. The narrower pair returns actual astrophotography.
    "night_sky":    ("THE NIGHT SKY",   "(starry OR astrophotography)", None, True),
    "waterfalls":   ("WATERFALLS",      "waterfall cascade", None, True),
    "canyons":      ("CANYONS",         "canyon gorge cliffs", None, True),
    "volcanoes":    ("VOLCANOES",       "(volcano OR volcanic OR eruption OR volcan)", None, True),
    "lakes":        ("LAKES",           "(lake OR lagoon OR lago)", None, True),
    "islands":      ("ISLANDS",         "(island OR archipelago OR isla)", None, True),
    "caves":        ("CAVES",           "(cave OR cavern OR grotto OR cueva)", None, True),
    "autumn":       ("AUTUMN",          "autumn foliage forest", None, True),
    "winter":       ("WINTER",          "snow winter landscape", None, True),
    "wildflowers":  ("WILDFLOWERS",     "(wildflower OR wildflowers OR meadow OR blossom)", None, True),
    "fjords":       ("THE FJORDS",      "fjord landscape", None, True),
    "rainforest":   ("THE RAINFOREST",  "(rainforest OR jungle OR canopy)", None, True),
    "clouds":       ("THE CLOUDS",      "clouds sky formation", None, True),
    "mist":         ("THE MIST",        "fog mist landscape morning", None, True),
    "savanna":      ("THE SAVANNA",     "savanna grassland plain", None, True),
    "hot_springs":  ("HOT SPRINGS",     "(geyser OR thermal OR hotspring)", None, True),
    "salt_flats":   ("SALT FLATS",      "(salar OR saline OR salt)", None, True),
    "rain":         ("THE RAIN",        "(rain OR raindrops OR rainy)", None, True),
    "tundra":       ("THE TUNDRA",      "(tundra OR permafrost OR arctic)", None, True),
    "ancient_trees":("ANCIENT TREES",   "(oak OR baobab OR sequoia OR adansonia OR quercus)", None, True),
    # --- creatures ---
    "birds":        ("BIRDS IN FLIGHT", "(bird OR birds OR aves)", None, False),
    "whales":       ("THE WHALES",      "(whale OR humpback OR orca OR megaptera OR balaenoptera)", None, False),
    "butterflies":  ("BUTTERFLIES",     "butterfly wings", None, False),
    "owls":         ("THE OWLS",        "(owl OR owls OR strix OR bubo OR tyto)", None, False),
    "foxes":        ("THE FOXES",       "fox wildlife", None, False),
    "deer":         ("THE DEER",        "deer forest wildlife", None, False),
    "penguins":     ("THE PENGUINS",    "(penguin OR penguins OR pygoscelis OR spheniscus)", None, False),
    "jellyfish":    ("JELLYFISH",       "(jellyfish OR chrysaora OR aurelia OR cyanea OR rhizostoma)", None, False),
    "reefs":        ("CORAL REEFS",     "coral reef underwater fish", None, False),
    "big_cats":     ("THE GREAT CATS",  "(leopard OR tiger OR lion OR jaguar OR cheetah OR panthera)", None, False),
    "elephants":    ("THE ELEPHANTS",   "(elephant OR elephants OR loxodonta OR elephas)", None, False),
    "horses":       ("THE HORSES",      "(horse OR horses OR stallion OR equus)", None, False),
    "wolves":       ("THE WOLVES",      "(wolf OR wolves)", None, False),
    "hummingbirds": ("HUMMINGBIRDS",    "(hummingbird OR hummingbirds OR trochilidae OR colibri)", None, False),
    "sea_turtles":  ("SEA TURTLES",     "sea turtle underwater", None, False),
    "dragonflies":  ("DRAGONFLIES",     "(dragonfly OR dragonflies OR damselfly OR odonata OR libellula)", None, False),
    # --- the heavens ---
    # Both forms spelled out: "galaxies" is an irregular plural, which the
    # relevance guard deliberately does not try to infer.
    "galaxies":     ("THE GALAXIES",    "(galaxy OR galaxies)", "incategory:Featured_pictures_of_astronomy", False),
    "nebulae":      ("THE NEBULAE",     "nebula", "incategory:Featured_pictures_of_astronomy", False),
    # No "luna": it matches Chile's Valle de la Luna, which is a desert.
    "the_moon":     ("THE MOON",        "(moon OR lunar)", "incategory:Featured_pictures_of_astronomy", False),
    "planets":      ("THE PLANETS",     "(saturn OR jupiter OR neptune OR planet)", "incategory:Featured_pictures_of_astronomy", False),
    "star_clusters":("THE STARS",       "(cluster OR globular OR pleiades)", "incategory:Featured_pictures_of_astronomy", False),
    "eclipses":     ("THE ECLIPSE",     "(eclipse OR totality OR annular)", "incategory:Featured_pictures_of_astronomy", False),
    # --- the very small — detail you have to lean in for ---
    "snowflakes":   ("SNOWFLAKES",        "(snowflake OR snowflakes)", None, False),
    "frost":        ("THE FROST",         "(frost OR hoarfrost OR rime)", None, False),
    "dew":          ("THE DEW",           "(dew OR dewdrop OR dewdrops)", None, False),
    "water_drops":  ("WATER DROPS",       "(droplet OR droplets OR waterdrop)", None, False),
    "soap_bubbles": ("BUBBLES",           "(bubble OR bubbles)", None, False),
    "spider_webs":  ("SPIDER WEBS",       "(cobweb OR spiderweb OR web)", None, False),
    "feathers":     ("FEATHERS",          "(feather OR feathers OR plumage)", None, False),
    "seashells":    ("SEASHELLS",         "(seashell OR shell OR conch)", None, False),
    "crystals":     ("CRYSTALS",          "(crystal OR crystals OR quartz)", None, False),
    "minerals":     ("THE MINERALS",      "(mineral OR minerals OR malachite)", None, False),
    "geodes":       ("GEODES",            "(geode OR agate OR amethyst)", None, False),
    "amber":        ("AMBER",             "(amber OR resin)", None, False),
    "leaf_veins":   ("THE LEAF",          "(leaf OR leaves OR foliage)", None, False),
    "tree_rings":   ("TREE RINGS",        "(dendrochronology OR treering OR trunk)", None, False),
    "honeycomb":    ("THE HONEYCOMB",     "(honeycomb OR beehive OR honeybee)", None, False),
    "moss":         ("THE MOSS",          "(moss OR mosses OR bryophyte)", None, False),
    "lichen":       ("THE LICHEN",        "(lichen OR lichens)", None, False),
    "fungi":        ("THE FUNGI",         "(mushroom OR fungus OR fungi)", None, False),
    "seeds":        ("SEEDS",             "(seed OR seeds OR dandelion)", None, False),
    "ice_crystals": ("ICE",               "(icicle OR icicles OR ice)", None, False),
    "pebbles":      ("THE STONES",        "(pebble OR pebbles OR shingle)", None, False),
    "bark":         ("THE BARK",          "(bark OR trunk)", None, False),
    # --- growing things ---
    "orchids":      ("THE ORCHIDS",       "(orchid OR orchids OR orchidaceae)", None, False),
    "lotus":        ("THE LOTUS",         "(lotus OR nelumbo OR waterlily)", None, False),
    "cacti":        ("THE CACTI",         "(cactus OR cacti OR saguaro)", None, False),
    "succulents":   ("SUCCULENTS",        "(succulent OR succulents OR aloe)", None, False),
    "ferns":        ("THE FERNS",         "(fern OR ferns OR frond)", None, False),
    "redwoods":     ("THE REDWOODS",      "(redwood OR sequoia OR sequoiadendron)", None, True),
    "palms":        ("THE PALMS",         "(palm OR palms OR palmera)", None, True),
    "cherry_blossom":("THE BLOSSOM",       "(sakura OR cherry OR blossom)", None, True),
    "lavender":     ("LAVENDER",          "(lavender OR lavandula)", None, True),
    "poppies":      ("THE POPPIES",       "(poppy OR poppies OR papaver)", None, True),
    "bluebells":    ("BLUEBELLS",         "(bluebell OR bluebells OR hyacinthoides)", None, True),
    "sunflowers":   ("SUNFLOWERS",        "(sunflower OR sunflowers OR helianthus)", None, True),
    "wisteria":     ("WISTERIA",          "(wisteria OR glycine)", None, False),
    "bamboo":       ("THE BAMBOO",        "(bamboo OR bambusa)", None, True),
    "kelp":         ("THE KELP",          "(kelp OR seaweed OR algae)", None, False),
    "tulips":       ("THE TULIPS",        "(tulip OR tulips OR tulipa)", None, True),
    "roses":        ("THE ROSES",         "(rose OR roses OR rosa)", None, False),
    "water_lilies": ("WATER LILIES",      "(waterlily OR nymphaea OR lily)", None, False),
    "heather":      ("THE HEATHER",       "(heather OR calluna OR heath)", None, True),
    "rice_terraces":("THE TERRACES",      "(terrace OR terraces OR paddy)", None, True),
    "vineyards":    ("THE VINEYARDS",     "(vineyard OR vineyards OR vine)", None, True),
    "olive_groves": ("THE OLIVE TREES",   "(olive OR olea)", None, True),
    "tea_fields":   ("THE TEA FIELDS",    "(tea OR camellia OR plantation)", None, True),
    # --- more creatures ---
    "flamingos":    ("THE FLAMINGOS",     "(flamingo OR flamingos OR phoenicopterus)", None, False),
    "herons":       ("THE HERONS",        "(heron OR herons OR ardea)", None, False),
    "cranes":       ("THE CRANES",        "(crane OR cranes OR grus)", None, False),
    "eagles":       ("THE EAGLES",        "(eagle OR eagles OR aquila OR haliaeetus)", None, False),
    "kingfishers":  ("KINGFISHERS",       "(kingfisher OR alcedo)", None, False),
    "puffins":      ("THE PUFFINS",       "(puffin OR puffins OR fratercula)", None, False),
    "peacocks":     ("THE PEACOCKS",      "(peacock OR peafowl OR pavo)", None, False),
    "parrots":      ("THE PARROTS",       "(parrot OR parrots OR macaw OR ara)", None, False),
    "swans":        ("THE SWANS",         "(swan OR swans OR cygnus)", None, False),
    "pelicans":     ("THE PELICANS",      "(pelican OR pelicans OR pelecanus)", None, False),
    "storks":       ("THE STORKS",        "(stork OR storks OR ciconia)", None, False),
    "toucans":      ("THE TOUCANS",       "(toucan OR ramphastos)", None, False),
    "woodpeckers":  ("WOODPECKERS",       "(woodpecker OR picus OR dendrocopos)", None, False),
    "ravens":       ("THE RAVENS",        "(raven OR crow OR corvus)", None, False),
    "bees":         ("THE BEES",          "(bee OR bees OR bombus OR apis)", None, False),
    "beetles":      ("THE BEETLES",       "(beetle OR beetles OR coleoptera)", None, False),
    "ladybirds":    ("LADYBIRDS",         "(ladybird OR ladybug OR coccinella)", None, False),
    "moths":        ("THE MOTHS",         "(moth OR moths OR sphingidae)", None, False),
    "ants":         ("THE ANTS",          "(ant OR ants OR formica)", None, False),
    "mantises":     ("THE MANTIS",        "(mantis OR mantid)", None, False),
    "spiders":      ("THE SPIDERS",       "(spider OR spiders OR araneae)", None, False),
    "snails":       ("THE SNAILS",        "(snail OR snails OR helix)", None, False),
    "frogs":        ("THE FROGS",         "(frog OR frogs OR rana OR hyla)", None, False),
    "chameleons":   ("CHAMELEONS",        "(chameleon OR chamaeleo)", None, False),
    "geckos":       ("THE GECKOS",        "(gecko OR geckos)", None, False),
    "iguanas":      ("THE IGUANAS",       "(iguana OR iguanas)", None, False),
    "snakes":       ("THE SERPENTS",      "(snake OR snakes OR serpent)", None, False),
    "crocodiles":   ("CROCODILES",        "(crocodile OR alligator OR crocodylus)", None, False),
    "dolphins":     ("THE DOLPHINS",      "(dolphin OR dolphins OR delphinus)", None, False),
    "sharks":       ("THE SHARKS",        "(shark OR sharks OR carcharhinus)", None, False),
    "rays":         ("THE RAYS",          "(stingray OR manta OR ray)", None, False),
    "seahorses":    ("SEAHORSES",         "(seahorse OR hippocampus)", None, False),
    "octopus":      ("THE OCTOPUS",       "(octopus OR cuttlefish OR squid)", None, False),
    "starfish":     ("THE STARFISH",      "(starfish OR seastar OR asteroidea)", None, False),
    "crabs":        ("THE CRABS",         "(crab OR crabs OR brachyura)", None, False),
    "nudibranchs":  ("SEA SLUGS",         "(nudibranch OR nudibranchs)", None, False),
    "otters":       ("THE OTTERS",        "(otter OR otters OR lutra)", None, False),
    "seals":        ("THE SEALS",         "(seal OR seals OR pinniped)", None, False),
    "bears":        ("THE BEARS",         "(bear OR bears OR ursus)", None, False),
    "pandas":       ("THE PANDAS",        "(panda OR ailuropoda)", None, False),
    "monkeys":      ("THE MONKEYS",       "(monkey OR macaque OR primate)", None, False),
    "lemurs":       ("THE LEMURS",        "(lemur OR lemurs OR lemuridae)", None, False),
    "sloths":       ("THE SLOTHS",        "(sloth OR bradypus)", None, False),
    "koalas":       ("THE KOALAS",        "(koala OR phascolarctos)", None, False),
    "kangaroos":    ("KANGAROOS",         "(kangaroo OR wallaby OR macropus)", None, False),
    "giraffes":     ("THE GIRAFFES",      "(giraffe OR giraffa)", None, False),
    "zebras":       ("THE ZEBRAS",        "(zebra OR zebras OR equus)", None, False),
    "rhinos":       ("THE RHINOS",        "(rhinoceros OR rhino OR ceratotherium)", None, False),
    "hippos":       ("THE HIPPOS",        "(hippopotamus OR hippo)", None, False),
    "bison":        ("THE BISON",         "(bison OR buffalo)", None, False),
    "camels":       ("THE CAMELS",        "(camel OR camels OR camelus)", None, False),
    "ibex":         ("THE IBEX",          "(ibex OR capra OR chamois)", None, False),
    "squirrels":    ("SQUIRRELS",         "(squirrel OR sciurus)", None, False),
    "hedgehogs":    ("HEDGEHOGS",         "(hedgehog OR erinaceus)", None, False),
    "hares":        ("THE HARES",         "(hare OR rabbit OR lepus)", None, False),
    "lynx":         ("THE LYNX",          "(lynx OR bobcat)", None, False),
    "moose":        ("THE MOOSE",         "(moose OR elk OR alces)", None, False),
    "reindeer":     ("THE REINDEER",      "(reindeer OR caribou OR rangifer)", None, False),
    "bats":         ("THE BATS",          "(bat OR bats OR chiroptera)", None, False),
    "salmon":       ("THE SALMON",        "(salmon OR salmo OR trout)", None, False),
    # --- weather and light ---
    "rainbows":     ("THE RAINBOW",       "(rainbow OR rainbows)", None, True),
    "sunbeams":     ("THE LIGHT",         "(crepuscular OR sunbeam OR sunrays)", None, True),
    "mammatus":     ("THE SKY",           "(mammatus OR undulatus)", None, True),
    "noctilucent":  ("NIGHT CLOUDS",      "(noctilucent OR polar)", None, True),
    "sandstorms":   ("THE SANDSTORM",     "(sandstorm OR haboob OR dust)", None, True),
    # --- the shape of the earth ---
    "sea_stacks":   ("SEA STACKS",        "(stack OR stacks OR seastack)", None, True),
    "arches":       ("THE ARCHES",        "(arch OR arches OR natural)", None, True),
    "hoodoos":      ("THE HOODOOS",       "(hoodoo OR hoodoos OR pinnacle)", None, True),
    "karst":        ("THE KARST",         "(karst OR limestone)", None, True),
    "badlands":     ("THE BADLANDS",      "(badlands OR badland)", None, True),
    "mangroves":    ("THE MANGROVES",     "(mangrove OR mangroves OR rhizophora)", None, True),
    "wetlands":     ("THE WETLANDS",      "(wetland OR marsh OR swamp)", None, True),
    "deltas":       ("THE DELTA",         "(delta OR estuary)", None, True),
    "moorland":     ("THE MOORS",         "(moor OR moorland OR heathland)", None, True),
    "steppe":       ("THE STEPPE",        "(steppe OR prairie OR grassland)", None, True),
    "frozen_lakes": ("FROZEN WATER",      "(frozen OR icebound)", None, True),
    "ice_caves":    ("ICE CAVES",         "(icecave OR glacier OR crevasse)", None, True),
    "lava_flows":   ("THE LAVA",          "(lava OR magma OR basalt)", None, True),
    "fumaroles":    ("THE VENTS",         "(fumarole OR solfatara OR mudpot)", None, True),
    "atolls":       ("THE ATOLLS",        "(atoll OR lagoon OR reef)", None, True),
    "mesas":        ("THE MESAS",         "(mesa OR butte OR plateau)", None, True),
    "beaches":      ("THE SHORE",         "(beach OR shore OR coastline)", None, True),
    "cliffs":       ("THE CLIFFS",        "(cliff OR cliffs OR escarpment)", None, True),
    "springs":      ("THE SPRINGS",       "(spring OR source OR fuente)", None, True),
    # --- the very large ---
    "comets":       ("THE COMETS",        "(comet OR comets)", "incategory:Featured_pictures_of_astronomy", False),
    "meteors":      ("THE METEORS",       "(meteor OR perseid OR perseids)", "incategory:Featured_pictures_of_astronomy", False),
    "star_trails":  ("STAR TRAILS",       "(startrail OR startrails OR circumpolar)", None, False),
    "the_sun":      ("THE SUN",           "(sun OR solar OR prominence)", "incategory:Featured_pictures_of_astronomy", False),
    "supernovae":   ("SUPERNOVAE",        "(supernova OR remnant)", "incategory:Featured_pictures_of_astronomy", False),
    "star_birth":   ("WHERE STARS FORM",  "(pillars OR protostar OR stellar)", "incategory:Featured_pictures_of_astronomy", False),
    "andromeda":    ("ANDROMEDA",         "(andromeda OR messier)", "incategory:Featured_pictures_of_astronomy", False),
    "saturn_rings": ("THE RINGS",         "(saturn OR rings OR cassini)", "incategory:Featured_pictures_of_astronomy", False),
    "jupiter":      ("JUPITER",           "(jupiter OR jovian OR juno)", "incategory:Featured_pictures_of_astronomy", False),
    "mars":         ("MARS",              "(mars OR martian)", "incategory:Featured_pictures_of_astronomy", False),
    "earth_space":  ("OUR OWN EARTH",     "(earth OR terra OR blue)", "incategory:Featured_pictures_of_astronomy", False),
}


# The cover is the only slide most people see, so its shape rotates. A single
# headline template is recognisable within about four posts and gets scrolled
# past; numbers and questions keep the curiosity gap open.
COVER_VARIANTS = [
    "THE CREATIVITY OF GOD IN {noun}",
    "{count} PLACES THAT LOOK PAINTED",
    "DID GOD OVERDO IT WITH {noun}?",
    "{noun}, AND NOTHING WE MADE",
    "{count} THINGS NOBODY DESIGNED",
    "YOU HAVE NEVER SEEN {noun} LIKE THIS",
]
# Dropped: "LOOK AT {noun}". It is an instruction, not a hook — it opens no
# curiosity gap, and it is the one that fired on the flattest cover the account
# has published. Every variant here either poses a question, promises a count,
# or makes a claim worth checking.
# Wildlife and space are not "places"; keep those variants off them.
PLACE_ONLY_VARIANTS = {1}

# Comments are the strongest ranking signal Instagram has. These are
# deliberately answerable in two words — effort is what kills reply rates.
QUESTIONS = [
    "Which one would you stand in?",
    "Which slide made you stop?",
    "Save this for the day you need it — which one?",
    "Where would you take this in?",
    "Tag someone who needs to see slide 3.",
]

# The closing slide asks for the SAVE, not the follow. Saves are weighted x12
# and shares x20 in hashtags.SCORE_WEIGHTS, a follow is not weighted at all, and
# "follow for more" is the ask every account on the platform is already making.
#
# It also no longer states a cadence. The old copy promised "twice a week" while
# the account published a carousel daily — a brand promise contradicted by the
# posting schedule is worse than no promise, and it silently goes stale every
# time PLAN changes.
CTA_LINES = ("keep this one", "for a slower morning")
CTA_MICRO = "SAVE  ·  SHARE  ·  FOLLOW"


def _cfg(key: str, default: str) -> str:
    value = config.app.get(key, default)
    return default if value in (None, "") else str(value)


def wordmark() -> str:
    return _cfg("brand_wordmark", "holy ordinary")


def tagline() -> tuple[str, str]:
    raw = _cfg("brand_tagline", "creation × wonder")
    parts = [p.strip() for p in re.split(r"[×x]", raw, maxsplit=1)]
    return (parts[0], "× " + parts[1]) if len(parts) == 2 else (raw, "")


def _furniture_scrim(img: Image.Image) -> Image.Image:
    """Soft top/bottom gradients so the small furniture text stays readable
    on any photograph, without flattening the image the way a full scrim does."""
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    band = int(h * 0.22)
    for y in range(band):
        a = int(120 * (1 - y / band) ** 1.5)
        od.line([(0, y), (w, y)], fill=(0, 0, 0, a))
        od.line([(0, h - 1 - y), (w, h - 1 - y)], fill=(0, 0, 0, int(a * 1.05)))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _draw_furniture(img: Image.Image, location: Optional[str], credit: Optional[str],
                    mark_at_top: bool) -> Image.Image:
    w, h = img.size
    draw = ImageDraw.Draw(img)

    mark_size = int(w * 0.026)      # serif needs more size than a sans to read
    small_size = int(w * 0.0135)
    f_mark = ty.font(ty.SERIF, mark_size, "Light")
    f_small = ty.font(ty.SANS, small_size, "Light")

    # Wordmark and tagline swap ends between slides, as the reference does —
    # it keeps a long carousel from feeling like the same frame repeated.
    mark_y = int(h * 0.055) if mark_at_top else int(h * 0.905)
    tag_y = int(h * 0.90) if mark_at_top else int(h * 0.055)

    ty.draw_centered(draw, w, mark_y, wordmark(), f_mark, (255, 255, 255, 240),
                     ty.TRACK_WORDMARK)

    t1, t2 = tagline()
    for i, line in enumerate([t1, t2]):
        if not line:
            continue
        ty.draw_centered(draw, w, tag_y + i * int(small_size * 1.6), line.upper(),
                         f_small, (255, 255, 255, 200), ty.TRACK_MICRO)

    margin = int(w * 0.062)
    if location:
        for i, line in enumerate(location.split("\n")[:2]):
            ty.draw_tracked(draw, (margin, int(h * 0.845) + i * int(small_size * 1.5)),
                            line.upper(), f_small, (255, 255, 255, 230), ty.TRACK_MICRO)
    if credit:
        # CC BY requires attribution; keep it discreet but present on-slide.
        f_credit = ty.font(ty.SANS, int(w * 0.0105), "Regular")
        credit = _fit_credit(draw, credit, f_credit, w - 2 * margin)
        cw = ty.width(draw, credit, f_credit, ty.TRACK_MICRO)
        ty.draw_tracked(draw, (w - margin - cw, int(h * 0.868)), credit, f_credit,
                        (255, 255, 255, 145), ty.TRACK_MICRO)
    return img


def _fit_credit(draw, credit: str, font, max_px: int) -> str:
    """Shorten an over-long credit so it cannot run off the slide.

    The credit is right-aligned by measuring its own width, so nothing stopped a
    long one starting at a negative x. Observatory images are the case that
    breaks it: the Tarantula Nebula cover published 2026-08-15 credits eleven
    named astronomers and their institutions, and the line ran off BOTH edges of
    the frame.

    Truncation is on the AUTHOR only and the licence is always kept, because the
    licence is the part with legal weight — CC BY requires attribution, and "et
    al." is accepted scholarly practice for a long author list where dropping the
    licence would not be acceptable at all.
    """
    if ty.width(draw, credit, font, ty.TRACK_MICRO) <= max_px:
        return credit
    author, _, licence = credit.rpartition(" / ")
    if not author:
        return credit
    words = author.split()
    while words and ty.width(draw, f"{' '.join(words)} et al. / {licence}",
                             font, ty.TRACK_MICRO) > max_px:
        words.pop()
    if not words:
        return licence
    return f"{' '.join(words)} et al. / {licence}"


def _cover_slide(photo_img: Image.Image, title: str) -> Image.Image:
    img = _cover(photo_img, WIDTH, HEIGHT)
    img = ImageEnhance.Color(img).enhance(0.9)
    img = img.filter(ImageFilter.GaussianBlur(radius=WIDTH * 0.0015))
    # The cover carries a large title, so it needs a real scrim, unlike the
    # interior slides which only carry small furniture.
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 90))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    img = _furniture_scrim(img)

    draw = ImageDraw.Draw(img)
    max_w = int(WIDTH * 0.90)
    fnt, lines = ty.fit(draw, title, ty.DISPLAY, max_w, int(HEIGHT * 0.27),
                        start=int(WIDTH * 0.068), min_size=int(WIDTH * 0.036),
                        instance="Bold", tracking=ty.TRACK_TITLE, leading=ty.LEAD_TITLE)
    line_h = int(fnt.size * ty.LEAD_TITLE)
    y = int(HEIGHT * 0.50) - (len(lines) * line_h) // 2
    for line in lines:
        # No drop shadow: the scrim already guarantees contrast, and a hard
        # offset shadow is the fastest way to make display type look cheap.
        ty.draw_centered(draw, WIDTH, y, line, fnt, (255, 255, 255), ty.TRACK_TITLE)
        y += line_h
    return img


def _recent_subjects(limit: Optional[int] = None) -> list:
    """Subjects already published, oldest first.

    The window must be at least as long as the subject pool, or the rotation
    starts repeating while subjects that have never run are still waiting.
    """
    import json
    try:
        with open(os.path.join(OUT_DIR, "recent_subjects.json"), encoding="utf-8") as fh:
            history = list(json.load(fh))
            return history[-limit:] if limit else history
    except (OSError, ValueError):
        return []


def _remember_subject(subject: str, keep: int = 500) -> None:
    import json
    os.makedirs(OUT_DIR, exist_ok=True)
    recent = [s for s in _recent_subjects(keep) if s != subject]
    recent.append(subject)
    try:
        with open(os.path.join(OUT_DIR, "recent_subjects.json"), "w", encoding="utf-8") as fh:
            json.dump(recent[-keep:], fh)
    except OSError as exc:
        logger.warning(f"could not persist recent subjects: {exc}")


def rank_subjects(pool: Optional[list] = None) -> list:
    """Candidate subjects, least-recently-used first.

    Never-published subjects come first (shuffled among themselves), then the
    rest oldest-first. Callers that try several subjects in order therefore work
    through the whole pool before any topic comes round again — the account had
    published `mountains` three times while 45 subjects had never run once.
    """
    candidates = [s for s in (pool if pool is not None else SUBJECTS) if s in SUBJECTS]
    history = _recent_subjects()
    last_used = {subject: i for i, subject in enumerate(history)}
    unused = [s for s in candidates if s not in last_used]
    random.shuffle(unused)
    used = sorted((s for s in candidates if s in last_used), key=lambda s: last_used[s])
    return unused + used


def choose_subject() -> str:
    """Least-recently-used, so the pool is worked through before repeating."""
    ranked = rank_subjects()
    return ranked[0] if ranked else random.choice(list(SUBJECTS))


def _used_photos_path() -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, "used_photos.json")


def _recent_photo_urls(limit: int = 300) -> set:
    try:
        import json
        with open(_used_photos_path(), encoding="utf-8") as fh:
            return set(list(json.load(fh))[-limit:])
    except (OSError, ValueError):
        return set()


def _remember_photos(urls: list, keep: int = 300) -> None:
    import json
    existing = [u for u in _recent_photo_urls(keep) if u]
    existing.extend(u for u in urls if u not in existing)
    try:
        with open(_used_photos_path(), "w", encoding="utf-8") as fh:
            json.dump(existing[-keep:], fh)
    except OSError as exc:
        logger.warning(f"could not persist used photos: {exc}")


def _cta_slide(photo_img: Image.Image) -> Image.Image:
    """Closing slide. People who swiped this far are the warmest audience the
    account will ever have, and until now they were shown a photo and nothing
    to do."""
    img = _cover(photo_img, WIDTH, HEIGHT)
    # Softened, not obliterated. The old treatment stacked a 17px blur, a 0.55
    # desaturation AND a 150-alpha black wash on the same frame, which turned a
    # photograph into grey mush and made the last thing a swiper saw the worst
    # image in the set. The type here is large and bold; it needs far less help
    # than the verse cards do, and the slide should still read as the cover it
    # bookends.
    img = ImageEnhance.Color(img).enhance(0.72)
    img = img.filter(ImageFilter.GaussianBlur(radius=WIDTH * 0.005))
    img = Image.alpha_composite(img.convert("RGBA"),
                                Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 105))).convert("RGB")
    draw = ImageDraw.Draw(img)

    f_lead = ty.font(ty.SERIF, int(WIDTH * 0.058), "Light")
    y = int(HEIGHT * 0.40)
    for line in CTA_LINES:
        ty.draw_centered(draw, WIDTH, y, line, f_lead, (255, 255, 255, 240), 0.02)
        y += int(f_lead.size * 1.28)

    f_mark = ty.font(ty.DISPLAY, int(WIDTH * 0.052), "Bold")
    ty.draw_centered(draw, WIDTH, y + int(HEIGHT * 0.035), wordmark().upper(), f_mark,
                     (255, 255, 255), ty.TRACK_TITLE)

    f_small = ty.font(ty.SANS, int(WIDTH * 0.017), "Light")
    ty.draw_centered(draw, WIDTH, y + int(HEIGHT * 0.11), CTA_MICRO, f_small,
                     (255, 255, 255, 205), ty.TRACK_MICRO)
    return img


def build(subject: Optional[str] = None, slides: int = 8,
          out_dir: str = OUT_DIR) -> Optional[dict]:
    """Build a carousel. Returns {paths, subject, title, photos, credits}."""
    subject = subject if subject in SUBJECTS else choose_subject()
    noun, query, extra_pool, label_locations = SUBJECTS[subject]
    slides = max(3, min(slides, MAX_SLIDES))

    found = wikimedia.search(query, limit=slides * 4, extra_pool=extra_pool)
    if len(found) < slides:
        logger.error(f"only {len(found)} usable photos for {subject!r}, need {slides}")
        return None

    # Spread across photographers so a carousel is not one person's portfolio,
    # and skip anything used in a recent carousel — otherwise the same striking
    # photograph reappears every few weeks.
    seen_before = _recent_photo_urls()
    fresh = [p for p in found if p.url not in seen_before]
    if len(fresh) >= slides:
        found = fresh
    else:
        logger.warning(f"only {len(fresh)} unseen photos for {subject!r}; allowing repeats")

    # Candidates in preference order — author-spread first, then everything
    # else as fallback for slides that turn out unusable at download time.
    ordered: list[wikimedia.Photo] = []
    per_author: dict[str, int] = {}
    for photo in found:
        key = photo.author.lower()[:40]
        if per_author.get(key, 0) >= 2:
            continue
        ordered.append(photo)
        per_author[key] = per_author.get(key, 0) + 1
    spread = {p.url for p in ordered}
    ordered += [p for p in found if p.url not in spread]

    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    paths, credits, scales = [], [], []
    used: list[wikimedia.Photo] = []
    cover_source: Optional[Image.Image] = None

    for photo in ordered:
        if len(paths) >= slides:
            break
        source = wikimedia.download(photo, target=(WIDTH, HEIGHT))
        if source is None:
            continue
        # Verify the pixels that actually arrived: the search filter judges the
        # original, and the thumbnail ceiling can still hand back less. Drop the
        # slide and take the next candidate rather than publish a soft one.
        scale = max(WIDTH / source.width, HEIGHT / source.height)
        if scale > MAX_UPSCALE:
            logger.warning(f"skipping {photo.title}: would upscale {scale:.2f}x")
            continue
        # Enough pixels is not the same as worth looking at. Judged on the source
        # before any furniture is drawn, and dropped for the next candidate the
        # same way an upscale is — the 191-subject pool is deep enough to afford
        # being picky, and a flat slide costs a swipe.
        looks_ok, why = quality.check_slide_aesthetics(source)
        if not looks_ok:
            logger.warning(f"skipping {photo.title}: {why}")
            continue
        index = len(paths)
        if index == 0:
            from app.services import hashtags as _hashtags

            choices = [v for i, v in enumerate(COVER_VARIANTS)
                       if label_locations or i not in PLACE_ONLY_VARIANTS]
            # Least-recently-used rather than random.choice: the cover is the one
            # slide most people ever see, and a uniform draw over six templates
            # repeats within a couple of posts.
            headline = _hashtags.rotate("cover_variant", choices).format(
                noun=noun, count=slides - 1)
            slide = _cover_slide(source, headline)
            cover_source = source        # the CTA bookends with it; don't refetch
        else:
            slide = _cover(source, WIDTH, HEIGHT)
            slide = _furniture_scrim(slide)
        slide = _draw_furniture(slide,
                                photo.location if label_locations else None,
                                wikimedia.credit_line(photo),
                                mark_at_top=(index % 2 == 0))
        path = os.path.join(out_dir, f"{stamp}-{subject}-{index:02d}.jpg")
        slide.save(path, "JPEG", quality=94, optimize=True, progressive=True)
        paths.append(path)
        used.append(photo)
        scales.append(scale)
        credits.append(f"{photo.location or 'unlisted'} — {wikimedia.credit_line(photo)}")

    if len(paths) < 3:
        logger.error("too few slides rendered")
        return None

    car = {"paths": paths, "subject": subject, "title": noun.title(),
           "photos": used, "credits": credits, "scales": scales}
    # Closing CTA slide, built from the cover image so the set bookends.
    if len(paths) >= 3 and cover_source is not None:
        cta_path = os.path.join(out_dir, f"{stamp}-{subject}-zz-cta.jpg")
        _cta_slide(cover_source).save(cta_path, "JPEG", quality=94, optimize=True)
        paths.append(cta_path)

    _remember_photos([p.url for p in used])
    _remember_subject(subject)
    ok, reason = quality.check_carousel(car)
    quality.log_result("carousel", ok, reason)
    if not ok:
        return None
    logger.info(f"carousel built: {len(paths)} slides for {subject}")
    return car


def science_note(subject: str) -> str:
    """A short, plain explanation of the phenomenon.

    The wonder-plus-science pairing is what makes this format work: the image
    carries the awe, the caption gives people something to learn and pass on.

    Deliberately asks for qualitative description and no statistics — a wrong
    figure is the kind of error a comment section corrects in public, and this
    text is LLM-written and unverified.
    """
    from app.services import llm

    prompt = (
        f"Write 2-3 short sentences explaining, in plain everyday English, how {subject} "
        "form or occur in nature. Be accurate and general. Do NOT include any numbers, "
        "statistics, dates, measurements or place names. No preamble, no title, no "
        "hashtags — just the sentences."
    )
    try:
        text = (llm._generate_response(prompt) or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"science note failed: {exc}")
        return ""
    text = " ".join(text.split())
    # Guard against the model ignoring the no-numbers instruction.
    if re.search(r"\d", text):
        logger.warning("science note contained figures; dropping it rather than risk a wrong one")
        return ""
    return text[:600]


# The first line is the one Instagram indexes for search, and it was the SAME
# sentence on every carousel ever published — the one text field that could
# carry new keywords was carrying none. The body below it was identical too, so
# two carousels in a row read as a form letter.
#
# {noun} is the plain subject ("mountains"), never the cover headline: the
# headline is a rotating hook and would render as "The creativity of God in did
# god overdo it with mountains?".
CAPTION_LEADS = [
    "The creativity of God in {noun}",
    "{noun}, and what they say about the One who made them",
    "A closer look at {noun}",
    "Photographs of {noun}, and a reason to slow down",
    "What {noun} are still doing without an audience",
]

CAPTION_BODIES = [
    "Creation keeps saying something we did not invent. "
    "Swipe through and let it slow you down for a minute.",
    "None of this was made to be photographed. It was here first, "
    "and it will be here after the scroll.",
    "Swipe slowly. There is no hurry in any of these.",
    "Every one of these is a real place, photographed by someone who went there.",
    "Worth looking at for longer than an algorithm expects you to.",
]

# Saves are weighted x12 and shares x20; nothing in the caption ever asked.
SAVE_ASKS = [
    "Save this set for a day that needs slowing down.",
    "Save it — and send it to whoever would stand there with you.",
    "Keep this one. Share it if it is not just for you.",
]


def build_caption(car: dict) -> tuple[str, str]:
    from app.services import hashtags

    set_id = hashtags.choose_set()
    noun = car["title"].lower()
    lead = hashtags.rotate("carousel_lead", CAPTION_LEADS).format(noun=noun)
    # Templates that OPEN with {noun} would otherwise start the caption — and so
    # the whole post — on a lowercase letter.
    lead = lead[:1].upper() + lead[1:]
    body = hashtags.rotate("carousel_body", CAPTION_BODIES)
    note = science_note(car["subject"].replace("_", " "))
    tags = " ".join(hashtags.tags_for(set_id))

    # No credit block here by choice: CC BY attribution is satisfied on-slide,
    # where every photographer is named beside their own image. Dropping the
    # caption list keeps it readable without breaching a licence.
    parts = [lead, body]
    if note:
        parts.append(note)
    parts.append(hashtags.rotate("carousel_question", QUESTIONS))
    parts.append(hashtags.rotate("carousel_save", SAVE_ASKS))
    parts.append(tags)
    return "\n\n".join(parts), set_id


def publish(car: dict, publish_at=None) -> dict:
    from app.services import hashtags
    from app.services.postiz import PostizService

    svc = PostizService()
    svc.post_type = "post"
    integration = svc.get_configured_integration()
    if not integration.get("success"):
        return integration
    if publish_at is None:
        selected = svc.select_publish_at(kind="carousel")
        if not selected.get("success"):
            return selected
        publish_at = selected.get("publish_at") or selected.get("date")

    media = []
    for path in car["paths"]:
        up = svc.upload_media(path)
        if not up.get("success"):
            return up
        media.append(up["media"])

    caption, set_id = build_caption(car)
    result = svc.schedule_post(media, caption, publish_at,
                               integration=integration["integration"],
                               kind="carousel", set_id=set_id)
    if result.get("success"):
        hashtags.mark_used(set_id)
    return result
