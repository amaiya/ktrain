import sys
import time

import testenv

from ktrain.text.summarization import TransformerSummarizer, LexRankSummarizer

ts = TransformerSummarizer()
ls = LexRankSummarizer()


sample_doc = """Archive-name: space/new_probes
Last-modified: $Date: 93/04/01 14:39:17 $

UPCOMING PLANETARY PROBES - MISSIONS AND SCHEDULES

    Information on upcoming or currently active missions not mentioned below
    would be welcome. Sources: NASA fact sheets, Cassini Mission Design
    team, ISAS/NASDA launch schedules, press kits.


    ASUKA (ASTRO-D) - ISAS (Japan) X-ray astronomy satellite, launched into
    Earth orbit on 2/20/93. Equipped with large-area wide-wavelength (1-20
    Angstrom) X-ray telescope, X-ray CCD cameras, and imaging gas
    scintillation proportional counters.


    CASSINI - Saturn orbiter and Titan atmosphere probe. Cassini is a joint
    NASA/ESA project designed to accomplish an exploration of the Saturnian
    system with its Cassini Saturn Orbiter and Huygens Titan Probe. Cassini
    is scheduled for launch aboard a Titan IV/Centaur in October of 1997.
    After gravity assists of Venus, Earth and Jupiter in a VVEJGA
    trajectory, the spacecraft will arrive at Saturn in June of 2004. Upon
    arrival, the Cassini spacecraft performs several maneuvers to achieve an
    orbit around Saturn. Near the end of this initial orbit, the Huygens
    Probe separates from the Orbiter and descends through the atmosphere of
    Titan. The Orbiter relays the Probe data to Earth for about 3 hours
    while the Probe enters and traverses the cloudy atmosphere to the
    surface. After the completion of the Probe mission, the Orbiter
    continues touring the Saturnian system for three and a half years. Titan
    synchronous orbit trajectories will allow about 35 flybys of Titan and
    targeted flybys of Iapetus, Dione and Enceladus. The objectives of the
    mission are threefold: conduct detailed studies of Saturn's atmosphere,
    rings and magnetosphere; conduct close-up studies of Saturn's
    satellites, and characterize Titan's atmosphere and surface.

    One of the most intriguing aspects of Titan is the possibility that its
    surface may be covered in part with lakes of liquid hydrocarbons that
    result from photochemical processes in its upper atmosphere. These
    hydrocarbons condense to form a global smog layer and eventually rain
    down onto the surface. The Cassini orbiter will use onboard radar to
    peer through Titan's clouds and determine if there is liquid on the
    surface. Experiments aboard both the orbiter and the entry probe will
    investigate the chemical processes that produce this unique atmosphere.

    The Cassini mission is named for Jean Dominique Cassini (1625-1712), the
    first director of the Paris Observatory, who discovered several of
    Saturn's satellites and the major division in its rings. The Titan
    atmospheric entry probe is named for the Dutch physicist Christiaan
    Huygens (1629-1695), who discovered Titan and first described the true
    nature of Saturn's rings.

	 Key Scheduled Dates for the Cassini Mission (VVEJGA Trajectory)
	 -------------------------------------------------------------
	   10/06/97 - Titan IV/Centaur Launch
	   04/21/98 - Venus 1 Gravity Assist
	   06/20/99 - Venus 2 Gravity Assist
	   08/16/99 - Earth Gravity Assist
	   12/30/00 - Jupiter Gravity Assist
	   06/25/04 - Saturn Arrival
	   01/09/05 - Titan Probe Release
	   01/30/05 - Titan Probe Entry
	   06/25/08 - End of Primary Mission
	    (Schedule last updated 7/22/92)


    GALILEO - Jupiter orbiter and atmosphere probe, in transit. Has returned
    the first resolved images of an asteroid, Gaspra, while in transit to
    Jupiter. Efforts to unfurl the stuck High-Gain Antenna (HGA) have
    essentially been abandoned. JPL has developed a backup plan using data
    compression (JPEG-like for images, lossless compression for data from
    the other instruments) which should allow the mission to achieve
    approximately 70% of its original objectives.

	   Galileo Schedule
	   ----------------
	   10/18/89 - Launch from Space Shuttle
	   02/09/90 - Venus Flyby
	   10/**/90 - Venus Data Playback
	   12/08/90 - 1st Earth Flyby
	   05/01/91 - High Gain Antenna Unfurled
	   07/91 - 06/92 - 1st Asteroid Belt Passage
	   10/29/91 - Asteroid Gaspra Flyby
	   12/08/92 - 2nd Earth Flyby
	   05/93 - 11/93 - 2nd Asteroid Belt Passage
	   08/28/93 - Asteroid Ida Flyby
	   07/02/95 - Probe Separation
	   07/09/95 - Orbiter Deflection Maneuver
	   12/95 - 10/97 - Orbital Tour of Jovian Moons
	   12/07/95 - Jupiter/Io Encounter
	   07/18/96 - Ganymede
	   09/28/96 - Ganymede
	   12/12/96 - Callisto
	   01/23/97 - Europa
	   02/28/97 - Ganymede
	   04/22/97 - Europa
	   05/31/97 - Europa
	   10/05/97 - Jupiter Magnetotail Exploration


    HITEN - Japanese (ISAS) lunar probe launched 1/24/90. Has made
    multiple lunar flybys. Released Hagoromo, a smaller satellite,
    into lunar orbit. This mission made Japan the third nation to
    orbit a satellite around the Moon.


    MAGELLAN - Venus radar mapping mission. Has mapped almost the entire
    surface at high resolution. Currently (4/93) collecting a global gravity
    map.


    MARS OBSERVER - Mars orbiter including 1.5 m/pixel resolution camera.
    Launched 9/25/92 on a Titan III/TOS booster. MO is currently (4/93) in
    transit to Mars, arriving on 8/24/93. Operations will start 11/93 for
    one martian year (687 days).


    TOPEX/Poseidon - Joint US/French Earth observing satellite, launched
    8/10/92 on an Ariane 4 booster. The primary objective of the
    TOPEX/POSEIDON project is to make precise and accurate global
    observations of the sea level for several years, substantially
    increasing understanding of global ocean dynamics. The satellite also
    will increase understanding of how heat is transported in the ocean.


    ULYSSES- European Space Agency probe to study the Sun from an orbit over
    its poles. Launched in late 1990, it carries particles-and-fields
    experiments (such as magnetometer, ion and electron collectors for
    various energy ranges, plasma wave radio receivers, etc.) but no camera.

    Since no human-built rocket is hefty enough to send Ulysses far out of
    the ecliptic plane, it went to Jupiter instead, and stole energy from
    that planet by sliding over Jupiter's north pole in a gravity-assist
    manuver in February 1992. This bent its path into a solar orbit tilted
    about 85 degrees to the ecliptic. It will pass over the Sun's south pole
    in the summer of 1993. Its aphelion is 5.2 AU, and, surprisingly, its
    perihelion is about 1.5 AU-- that's right, a solar-studies spacecraft
    that's always further from the Sun than the Earth is!

    While in Jupiter's neigborhood, Ulysses studied the magnetic and
    radiation environment. For a short summary of these results, see
    *Science*, V. 257, p. 1487-1489 (11 September 1992). For gory technical
    detail, see the many articles in the same issue.


    OTHER SPACE SCIENCE MISSIONS (note: this is based on a posting by Ron
    Baalke in 11/89, with ISAS/NASDA information contributed by Yoshiro
    Yamada (yamada@yscvax.ysc.go.jp). I'm attempting to track changes based
    on updated shuttle manifests; corrections and updates are welcome.

    1993 Missions
	o ALEXIS [spring, Pegasus]
	    ALEXIS (Array of Low-Energy X-ray Imaging Sensors) is to perform
	    a wide-field sky survey in the "soft" (low-energy) X-ray
	    spectrum. It will scan the entire sky every six months to search
	    for variations in soft-X-ray emission from sources such as white
	    dwarfs, cataclysmic variable stars and flare stars. It will also
	    search nearby space for such exotic objects as isolated neutron
	    stars and gamma-ray bursters. ALEXIS is a project of Los Alamos
	    National Laboratory and is primarily a technology development
	    mission that uses astrophysical sources to demonstrate the
	    technology. Contact project investigator Jeffrey J Bloch
	    (jjb@beta.lanl.gov) for more information.

	o Wind [Aug, Delta II rocket]
	    Satellite to measure solar wind input to magnetosphere.

	o Space Radar Lab [Sep, STS-60 SRL-01]
	    Gather radar images of Earth's surface.

	o Total Ozone Mapping Spectrometer [Dec, Pegasus rocket]
	    Study of Stratospheric ozone.

	o SFU (Space Flyer Unit) [ISAS]
	    Conducting space experiments and observations and this can be
	    recovered after it conducts the various scientific and
	    engineering experiments. SFU is to be launched by ISAS and
	    retrieved by the U.S. Space Shuttle on STS-68 in 1994.

    1994
	o Polar Auroral Plasma Physics [May, Delta II rocket]
	    June, measure solar wind and ions and gases surrounding the
	    Earth.

	o IML-2 (STS) [NASDA, Jul 1994 IML-02]
	    International Microgravity Laboratory.

	o ADEOS [NASDA]
	    Advanced Earth Observing Satellite.

	o MUSES-B (Mu Space Engineering Satellite-B) [ISAS]
	    Conducting research on the precise mechanism of space structure
	    and in-space astronomical observations of electromagnetic waves.

    1995
	LUNAR-A [ISAS]
	    Elucidating the crust structure and thermal construction of the
	    moon's interior.


    Proposed Missions:
	o Advanced X-ray Astronomy Facility (AXAF)
	    Possible launch from shuttle in 1995, AXAF is a space
	    observatory with a high resolution telescope. It would orbit for
	    15 years and study the mysteries and fate of the universe.

	o Earth Observing System (EOS)
	    Possible launch in 1997, 1 of 6 US orbiting space platforms to
	    provide long-term data (15 years) of Earth systems science
	    including planetary evolution.

	o Mercury Observer
	    Possible 1997 launch.

	o Lunar Observer
	    Possible 1997 launch, would be sent into a long-term lunar
	    orbit. The Observer, from 60 miles above the moon's poles, would
	    survey characteristics to provide a global context for the
	    results from the Apollo program.

	o Space Infrared Telescope Facility
	    Possible launch by shuttle in 1999, this is the 4th element of
	    the Great Observatories program. A free-flying observatory with
	    a lifetime of 5 to 10 years, it would observe new comets and
	    other primitive bodies in the outer solar system, study cosmic
	    birth formation of galaxies, stars and planets and distant
	    infrared-emitting galaxies

	o Mars Rover Sample Return (MRSR)
	    Robotics rover would return samples of Mars' atmosphere and
	    surface to Earch for analysis. Possible launch dates: 1996 for
	    imaging orbiter, 2001 for rover.

	o Fire and Ice
	    Possible launch in 2001, will use a gravity assist flyby of
	    Earth in 2003, and use a final gravity assist from Jupiter in
	    2005, where the probe will split into its Fire and Ice
	    components: The Fire probe will journey into the Sun, taking
	    measurements of our star's upper atmosphere until it is
	    vaporized by the intense heat. The Ice probe will head out
	    towards Pluto, reaching the tiny world for study by 2016."""

start = time.time()
print(ts.summarize(sample_doc))
print(time.time() - start)

start = time.time()
print(ls.summarize(sample_doc, num_sentences=5))
print(time.time() - start)
# sys.exit(1)


# zsl
from ktrain.text.zsl import ZeroShotClassifier

zsl = ZeroShotClassifier(device="cuda", quantize=True)
topic_strings = ["politics", "elections", "sports", "films", "television"]
doc = "I am unhappy with decisions of the government and will definitely vote in 2020."
start = time.time()
print(zsl.predict(doc, topic_strings=topic_strings, include_labels=True))
print(time.time() - start)


# translation
from ktrain.text.translation import EnglishTranslator

translator = EnglishTranslator(src_lang="zh", device=None, quantize=True)
src_text = """大流行对世界经济造成了严重破坏。但是，截至2020年6月，美国股票市场持续上涨。"""
start = time.time()
print(translator.translate(src_text))
print(time.time() - start)


# transcription
from ktrain.text.speech import Transcriber

transcriber = Transcriber()
afiles = ["resources/text_data/sample.wav"]
start = time.time()
result = transcriber.transcribe(afiles)
print(time.time() - start)
print(result)


# image-captioning
from ktrain.vision.caption import ImageCaptioner

ic = ImageCaptioner()
ifiles = ["resources/image_data/squirrel.jpg"]
start = time.time()
result = ic.caption(ifiles)
print(time.time() - start)
print(result)


# sentiment-analysis
from ktrain.text.sentiment import SentimentAnalyzer

classifier = SentimentAnalyzer()
start = time.time()
result = classifier.predict("I got a promotion today.")
print(time.time() - start)
print(result)
