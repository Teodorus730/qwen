# Real HF benchmark v1 pseudo-gold audit

## Purpose

This audit checks the pseudo-gold labels and evaluation logic before treating the real HF benchmark v1 metrics as meaningful. It does not tune classifiers, thresholds, or taxonomy labels.

## Schema sanity check

- Records: 120
- Missing required fields: {'expected_source_type': 0, 'expected_domain': 0, 'expected_field': 0, 'review_note': 0, 'review_confidence': 0}
- Review confidence range: 0.6 to 1.0
- Bad review confidence values: 0
- Unique expected_source_type values: ['boilerplate_or_noise', 'code', 'commercial_product', 'educational', 'forum_qa', 'legal_government', 'math', 'news', 'reference', 'web_general', 'wiki_reference']
- Unique expected_domain values: ['commercial', 'education', 'government', 'infrastructure', 'media', 'reference', 'science', 'software', 'stem', 'unknown']
- Unique expected_field values: ['biology', 'encyclopedic_article', 'environmental_science', 'general_education', 'legal_notice', 'mathematics', 'news', 'physics', 'product_page', 'programming', 'unknown', 'urban_systems']
- Unique expected_subfield values: [None, 'algebra', 'article', 'calculus', 'documentation', 'general', 'public_information', 'retail']
- Expected domain/field/subfield tuples outside taxonomy: 0
- Null expected_subfield count: 2; all are `unknown/unknown/null`.
- Duplicate pseudo-gold `chunk_id` values: 0

## Taxonomy gaps

- `patents/POS_systems`: 17
- `statistics`: 5
- `patents/hardware_networking`: 4
- `arithmetic/measurement`: 4
- `chemistry`: 3
- `art_history`: 2
- `sports/coaching`: 2
- `spam/low_quality_math`: 2
- `automotive`: 2
- `probability/statistics`: 2
- `geometry`: 2
- `food`: 2
- `topology`: 2
- `medicine/psychology`: 1
- `history/politics/international_relations`: 1

The biggest issue is not invalid labels: all pseudo-gold semantic tuples are taxonomy-valid. The issue is forced nearest-label mapping because real chunks contain many categories absent from the tiny MVP taxonomy.

## Evaluation correctness check

- Matching key: evaluator uses `chunk_id` from pseudo-gold and prediction files. This is correct for the current benchmark outputs.
- Dataset handling: evaluator reports per-dataset metrics from pseudo-gold records and does not mix datasets as long as `chunk_id` remains globally unique. Current chunk ids include dataset prefixes, so this is safe.
- Null handling: evaluator compares Python `None` directly. This correctly counts `null == null` for `unknown/unknown/null` cases and counts `null != article` when a classifier leaves a label empty.
- Field comparison: evaluator compares exact `source_type`, `domain`, `field`, and `subfield` values. This is strict but transparent.
- Taxonomy handling: evaluator does not validate taxonomy membership. This is acceptable here because pseudo-gold schema sanity found 0 out-of-taxonomy semantic tuples.
- Naming conventions: no source_type normalization is applied. This is mostly correct, but it means labels like `wiki_reference` vs `reference` or `web_general` vs `unknown` are strict mismatches by design.
- Possible issue: `load_by_chunk_id` silently overwrites duplicate `chunk_id` records if present. Current pseudo-gold and prediction files have no duplicate chunk ids, but duplicate detection would make the evaluator safer.
- No evaluator bug was found that would explain the low metrics. The low metrics mainly reflect strict exact-match scoring plus many null/low-confidence predictions.

## Spot-check records

- Spot-check entries: 40
- Unique spot-check chunk ids: 40
- Clearly questionable in spot-check: 3
- Low-confidence pseudo-gold labels in full file: 3

### A. Random records, seed 42

#### FineWeb-Edu_000000_004 (FineWeb-Edu)

- Text preview: Professor of Sociology and American Studies & Ethnicity at the University of Southern California where he also directs the Program for Environmental and Regional Equity and co-directs USC’s Center for the Study of Immigrant Integration. His most recent books include Just Growth: Inclusion and Prosperity in America’s Metropolitan Regions (Routledge 2012; co-authored with Chris Benner) Uncommon Common Ground: Race and America’s Future (W.W. Norton 2010; co-authored with Angela Glover Blackwell and...
- Expected: `reference/reference/encyclopedic_article/general`
- Review note: academic biography/publication list; missing_taxonomy: biography/social_science
- Review confidence: 0.8
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb_000009_027 (FineWeb)

- Text preview: At step S90, the "trapper" subroutine is executed. The trapper subroutine of step S90 is shown in detail in FIG. 4B. Referring now to FIG. 4B, the trapper subroutine starts at 125 and proceeds to step S130, in which a flag is detected, if it is present, indicating that video cassette rental data was being selected or "trapped" during the previous interrupt cycle. If the determination in step S130 is affirmative, processing proceeds to a series of steps S131-S134 for determining when the data on ...
- Expected: `reference/software/programming/documentation`
- Review note: patent algorithm/subroutine description; missing_taxonomy: patents/POS_systems
- Review confidence: 0.7
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb-Edu_000018_000 (FineWeb-Edu)

- Text preview: The OED “is widely regarded as the accepted authority on the English language. It is an unsurpassed guide to the meaning, history, and pronunciation of 600,000 words— past and present—from across the English-speaking world. As a historical dictionary, the OED is very different from those of current English, in which the focus is on present-day meanings. You’ll still find these in the OED, but you’ll also find the history of individual words, and of the language—traced through 3 million quotation...
- Expected: `reference/reference/encyclopedic_article/general`
- Review note: library database description for OED; language/reference resource
- Review confidence: 0.8
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/multilingual/mixed_language/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: pseudo_gold_seems_ok

#### FineMath_000009_000 (FineMath)

- Text preview: Linsolve - Maple Programming Help Home : Support : Online Help : Mathematics : Inert Functions : Linsolve Linsolve inert matrix solve Calling Sequence Linsolve(A, b) mod n Linsolve(A, b, 'r', 't') mod n Parameters A - rectangular Matrix b - Vector 'r' - (optional) name 't' - (optional) name n - an integer, the modulus Description
- Expected: `code/software/programming/documentation`
- Review note: Maple help page for Linsolve; math software documentation
- Review confidence: 0.9
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: pseudo_gold_seems_ok

#### FineMath_000012_002 (FineMath)

- Text preview: It is simple because the centre is too wide since the triangle, to assume a shape. That would be considered a triangle. sameday essay You can figure out the the negative lengths are corresponding to the span of this rectangle when you think of a triangle the shape of the parallelogram. The diameter of the rectangle is also still. It is not difficult to see that its edges are always going to be a little briefer than those sides After you believe of a triangle. It makes it seem like pointed. It is...
- Expected: `boilerplate_or_noise/unknown/unknown/null`
- Review note: low-quality generated/SEO text about triangles; missing_taxonomy: spam/low_quality_math
- Review confidence: 0.6
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_questionable; needs_human_review

#### FineWeb_000009_040 (FineWeb)

- Text preview: Upon pressing the function key F10 the transaction currently displayed on the screen is aborted. In accordance with the present invention, upon pressing the "hot keys" on the keyboard, that is, the "Alt" and the left "shift" keys, the main menu of the teacher program is displayed in the shaded area at the bottom of the display as shown in Screen 3 illustrated in FIG. 4G. The main menu allows the installer to select processing for either "rentals" or "returns". By pressing the "Alt" and "1" keys,...
- Expected: `reference/software/programming/documentation`
- Review note: technical software UI/patent procedure; missing_taxonomy: patents/POS_systems
- Review confidence: 0.7
- rule prediction: `educational/education/general_education/article`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `educational/education/general_education/article`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineMath_000026_007 (FineMath)

- Text preview: “We are assuming in this problem that the total force required is equally divided among the six ant feet, and that ONLY the contact between feet and ceiling gives rise to the force.” The weight of the ant is provided in Newtons (N), a derived unit which is the force needed to increase the speed of (or accelerate) one kilogram of mass one meter per second every second. A field test teacher passed around objects (e.g., a one Newton weight, an eight Newton cell phone) for students to be able to rel...
- Expected: `educational/science/physics/article`
- Review note: force/unit classroom explanation
- Review confidence: 0.9
- rule prediction: `educational/education/general_education/article`
- lexical prediction: `educational/science/physics/article`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `educational/education/general_education/article`
- Judgement: pseudo_gold_seems_ok

#### FineMath_000002_003 (FineMath)

- Text preview: expres the following as sums and differences of sines or cosines cos8t * sin2t sin(a+b) = sin(a)cos(b) + cos(a)sin(b) replacing by by -b and using that cos(-b)= cos(b) sin(-b)= -sin(b) gives: sin(a-b) = sin(a)cos(b) - cos(a)sin(b) … 3. ### math
- Expected: `math/stem/mathematics/algebra`
- Review note: trigonometry identities; closest taxonomy algebra
- Review confidence: 1.0
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: pseudo_gold_seems_ok

#### FineWeb-Edu_000000_000 (FineWeb-Edu)

- Text preview: Karuk Tribe: Learning from the First Californians for the Next California Editor's Note: This is part of series, Facing the Climate Gap, which looks at grassroots efforts in California low-income communities of color to address climate change and promote climate justice. This article was published in collaboration with GlobalPossibilities.org. The three sovereign entities in the United States are the federal government, the states and indigenous tribes, but according to Bill Tripp, a member of t...
- Expected: `news/science/environmental_science/article`
- Review note: climate justice article about Karuk Tribe; news/editorial style
- Review confidence: 0.9
- rule prediction: `educational/science/environmental_science/article`
- lexical prediction: `educational/science/environmental_science/article`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `educational/science/environmental_science/article`
- Judgement: pseudo_gold_seems_ok

#### FineWeb-Edu_000022_002 (FineWeb-Edu)

- Text preview: Picasso, Joan Miró, and Salvador Dalí talked on the radio in his place to publicize the show. Picasso's friend, the surrealist poet Paul Éluard, traveled down to Barcelona from Paris for the exhibit and delivered a lecture at the opening. Éluard, standing in for Picasso, received the hearty support of students who chanted, "Picasso, the Marxist," confirming that they saw Picasso as a radical. (In fact, although Picasso did later, in 1944, join the Communist party of France, Éluard, like most of ...
- Expected: `reference/reference/encyclopedic_article/general`
- Review note: historical art/politics narrative; missing_taxonomy: art_history/political_history
- Review confidence: 0.8
- rule prediction: `educational/education/general_education/article`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `educational/education/general_education/article`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

### B. Lowest review_confidence, excluding previous sections

#### FineMath_000012_001 (FineMath)

- Text preview: You may ask your self, how can we find out everything is a triangle in mathematics. In order to figure out a triangle, first we have http://en.wikipedia.com/wiki/Sigiriya to understand what it resembles. This really is a task that is harder because there are many unique shapes that could be created with all those 3 angles. Every shape has its own structures and rules. You may possibly consider just how to relate triangles to each of additional contours. As the exact length between your Valve’s v...
- Expected: `boilerplate_or_noise/unknown/unknown/null`
- Review note: low-quality SEO/generated text about triangles; missing_taxonomy: spam/low_quality_math
- Review confidence: 0.6
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_questionable; needs_human_review

#### FineWeb-Edu_000010_002 (FineWeb-Edu)

- Text preview: to look across the lake at the beauty that is unfolding. The scene is constant; the colors let me know that time is marching on. On the one hand I could worry that the seasons of my life are marching on, or, on the other, I could be challenged by the things I’ve learned this year that will help me to be wiser and more thoughtful in the future. One stunts natural growth; the other invigorates a sense of wonder about the world around us and the endless possibilities that potentially exist. The cho...
- Expected: `web_general/education/general_education/article`
- Review note: reflective life lesson/personal essay; missing_taxonomy: personal_essay
- Review confidence: 0.6
- rule prediction: `educational/education/general_education/article`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `educational/education/general_education/article`
- Judgement: taxonomy_gap; pseudo_gold_questionable; needs_human_review

#### FineWeb-Edu_000010_001 (FineWeb-Edu)

- Text preview: the sky. • In manhood, the adult relies on prayer and religious faith to sustain him through rough waters and a threatening landscape. • Finally, the man becomes old and the angel guides him to heaven across the waters of eternity. In each painting, accompanied by a guardian angel, the voyager rides the boat on the River of Life. The landscape, corresponding to the seasons of the year, plays a major role in telling the story. And in those paintings you can clearly see the leaves changing colors ...
- Expected: `educational/reference/encyclopedic_article/general`
- Review note: art/religious painting explanation; missing_taxonomy: art_history/religion
- Review confidence: 0.7
- rule prediction: `educational/education/general_education/article`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `educational/education/general_education/article`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb-Edu_000010_008 (FineWeb-Edu)

- Text preview: but clearly remembered moment in time. 9/11 does all of those things and more. My wife and I were in NYC: preparing to get on the George Washington Bridge to go into Manhattan when the first plane hit; coming to a complete stop on the road and in our lives; watching in fear and confusion as the second plane hit; staring in horror as first one and then the other building fell; hearing about the other plane crashes in Washington and Pennsylvania; staying glued to the radio and then the television ...
- Expected: `web_general/media/news/article`
- Review note: personal essay/reflection on 9/11; missing_taxonomy: personal_essay/history
- Review confidence: 0.7
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb-Edu_000010_010 (FineWeb-Edu)

- Text preview: in the USAF Manned Spaceflight Engineer Program. He was killed in the attacks of September 11, 2001 aboard American Airlines Flight 11, the first plane to hit the first World Trade Center building at 8:46am. All of the great values we read and write about seem to be interconnected, and loyalty may be the one at the hub of them all. Think of the people and things you’re loyal to, and then note the other great qualities that come from that loyalty. Friendship, success, pride, humility, professiona...
- Expected: `educational/education/general_education/article`
- Review note: values/loyalty reflective educational essay; contains biography/9-11 context
- Review confidence: 0.7
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: pseudo_gold_seems_ok

#### FineWeb-Edu_000010_011 (FineWeb-Edu)

- Text preview: tale begins at Camp Nazareth (that’s the name of the overnight camp at the end of our lake). Its run by the local Catholic Diocese which has had little success in recent years attracting enough kids. More often than not, this wonderful facility – it can hold up to 300 kids at any one time - is terribly under used. Fortunately, it seems that they’ve now discovered ways to attract alternate users like family reunions, corporate retreats and, just this past week, a high school crew team (Google “ro...
- Expected: `web_general/education/general_education/article`
- Review note: reflective anecdote about rowing/team learning; missing_taxonomy: sports/coaching
- Review confidence: 0.7
- rule prediction: `educational/education/general_education/article`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `educational/education/general_education/article`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb-Edu_000010_012 (FineWeb-Edu)

- Text preview: of the rowers is out of synch (even a little) the boat can very easily (and visibly) miss a beat. And if any of those misses are overly pronounced the boats can stop altogether or even capsize. So at the beginning of this training the coach definitely wanted to take it slow. As the week progressed, however, the boats began to move more smoothly, and over time they got smoother and faster. And since the object of crew is to beat the competition, smooth and fast is definitely better. In order to g...
- Expected: `educational/education/general_education/article`
- Review note: rowing/coaching lesson reflection; missing_taxonomy: sports/coaching
- Review confidence: 0.7
- rule prediction: `educational/education/general_education/article`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `educational/education/general_education/article`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb-Edu_000010_013 (FineWeb-Edu)

- Text preview: I watched this training unfold. Each of these young athletes was working hard to learn how to be the best they could be, they and their team mates were learning how to interact with each other more effectively, the coaches were seeing the results of their hard work and practice, and those of us on the sidelines were rewarded by seeing how things can and should work when effective instructions, practice and coaching all come together. We don’t often get to see things so clearly, or watch how the ...
- Expected: `web_general/education/general_education/article`
- Review note: rowing/teamwork reflective lesson; missing_taxonomy: sports/coaching/personal_essay
- Review confidence: 0.7
- rule prediction: `educational/education/general_education/article`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `educational/education/general_education/article`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb-Edu_000010_016 (FineWeb-Edu)

- Text preview: fashioned sayings? Here in New York last week the mayor and the meteorologists got it wrong – but not by much. The winds blew and the rains fell and, though there was less flooding and damage than predicted here, they made damn sure we were prepared by scaring the daylights out of us with their dire warnings. Now some people are complaining because they scared us; but those same people complained when they didn’t scare us before last winter’s massive snow storm, or that they didn’t scare others ...
- Expected: `web_general/science/environmental_science/article`
- Review note: weather/storm preparedness personal commentary; missing_taxonomy: weather/personal_blog
- Review confidence: 0.7
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb-Edu_000010_017 (FineWeb-Edu)

- Text preview: soul.” -Mark Twain Mark Twain achieved great success as a writer and public speaker. His wit and satire earned praise from critics and peers, and he was a friend to presidents, artists, industrialists, and European royalty. Loyalty can be both good and bad. People often remain loyal long after the reason for doing so has ended. If the reason you became loyal has petrified then you need to re-examine your motives and goals; you need to break free when the times demand it and it’s the right thing ...
- Expected: `web_general/education/general_education/article`
- Review note: reflective essay about loyalty with Mark Twain quote
- Review confidence: 0.7
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: pseudo_gold_seems_ok

### C. All classifiers disagree with pseudo-gold, excluding previous sections

#### FineWeb_000012_001 (FineWeb)

- Text preview: thoughts of death, including thoughts of suicide Where to Turn for Help If you have any of the symptoms of depression, get help. Your primary care physician is the best place to start, but not always the quickest. For help in finding a provider, click here If you are having suicidal thoughts, it is a medical emergency. Get help right away. In the Rochester area, contact Lifeline at (585) 275-5151, or get to the nearest hospital emergency department. The good news is that depression is very treat...
- Expected: `educational/science/biology/article`
- Review note: mental health help article; missing_taxonomy: medicine/psychology
- Review confidence: 0.8
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb-Edu_000014_000 (FineWeb-Edu)

- Text preview: Human nature and the nature of war Twenty years ago, Canadian military historian and journalist Gwynne Dyer fascinated the public in 45 countries with his award-winning television series on the nature of war. He followed up that success with an equally remarkable book that grew out of the research that went into the television series. Now Dr. Dyer has published, in his words, "a completely rewritten and updated new edition" of War. People who have never read the book should certainly do so, but ...
- Expected: `reference/reference/encyclopedic_article/general`
- Review note: book review / geopolitical reference; missing_taxonomy: history/politics/international_relations
- Review confidence: 0.8
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineMath_000018_003 (FineMath)

- Text preview: ###### Chapter 10. Experimental Design: Statistical Analysis of ... This is halfway between the 10th and 11th scores. Because both are 0, the median gain is 0. Similarly, the median gain on a running play is 3. The median is a particularly useful measure of central tendency when there are extreme scores at one end of a distribution. Such distributions are said to be skewed in the direction of the extreme scores. The median, unlike the mean, is unaffected by ... ###### A distribution is said to b...
- Expected: `educational/stem/mathematics/algebra`
- Review note: statistics lesson on median/skewed distributions; closest taxonomy math/algebra, missing_taxonomy: statistics
- Review confidence: 0.8
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineMath_000002_006 (FineMath)

- Text preview: Let y represent theta Prove: 1 + 1/tan^2y = 1/sin^2y My Answer: LS: = 1 + 1/tan^2y = (sin^2y + cos^2y) + 1 /(sin^2y/cos^2y) = (sin^2y + cos^2y) + 1 x (cos^2y/sin^2y) = (sin^2y + cos^2y) + (sin^2y + cos^2y) (cos^2y/sin^2y) = (sin^2y … 8. ### TRIG!
- Expected: `math/stem/mathematics/algebra`
- Review note: trigonometric identity proof; closest taxonomy algebra
- Review confidence: 1.0
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: pseudo_gold_seems_ok

#### FineWeb_000009_043 (FineWeb)

- Text preview: allows communication between the master computer and the data tap. In a preferred embodiment, both transmitted and received communications are differentially driven by utilizing RS-422 or RS-485 interfaces. The use of these interfaces allows effective communications over relatively large distances. That is, an RS-422 or RS-485 interface allows effective communication up to a length of approximately 5,000 feet, whereas a standard RS-232 interface typically restricts the effective length of a comm...
- Expected: `reference/software/programming/documentation`
- Review note: technical patent-style system description; missing_taxonomy: patents/hardware_networking
- Review confidence: 0.7
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineMath_000029_009 (FineMath)

- Text preview: A more serious limitation MAPE occurs when your data set can have 0 values. In the context of sports betting if you’re forecasting winning margins and a draw is possible then you can’t use MAPE. This is because if the final scores are level then Yi = 0 so PEi can’t be calculated due to a division by zero error. For this reason MAPE works best for modeling results such as total basketball, AFL and rugby scores rather than winning margins or football total scores, which can have zero values. In th...
- Expected: `educational/stem/mathematics/algebra`
- Review note: forecast error metric with division by zero; missing_taxonomy: statistics
- Review confidence: 0.8
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb_000009_051 (FineWeb)

- Text preview: the rental of the first video cassette. These steps are shown by displays 6 and 7 below. ##STR6## Next, the Total key 530 is pressed. As shown below, in Display 8, pressing this key causes the total amount, including tax, to be calculated and displayed. Further, individual subtotals of the rental charge; charges for any sales included in the transaction; and tax amounts also are displayed. ##STR7## Assume, at this point, that the customer gives the operator $5.00 in payment for the rentals. The ...
- Expected: `reference/software/programming/documentation`
- Review note: technical patent-style POS transaction flow; missing_taxonomy: patents/POS_systems
- Review confidence: 0.7
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb-Edu_000006_000 (FineWeb-Edu)

- Text preview: Great apes go through mid-life crisis They may not take up surfing or start second careers as cupcake-makers, but chimpanzees and orangutans seem to go through a ‘mid-life crisis’, just like humans. A study of 508 great apes in captivity shows that the animals’ sense of well-being bottoms out in their late 20s to mid-30s, the ape equivalent of middle age, before rebounding in old age. The finding that mid-life crises may not be uniquely human suggests that the events might have a biological, rat...
- Expected: `news/science/biology/article`
- Review note: science news about ape well-being study; missing_taxonomy: psychology/animal_behavior
- Review confidence: 0.8
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb_000006_003 (FineWeb)

- Text preview: Virtual hosting is only available in the professional edition Abyss Web Server X2. Abyss Web Server supports the SSL and TLS protocols and is able to accept secure connections with strong cryptography (up to 256-bit keys) to protect your visitors' sensitive data from flowing in clear form over the Internet. SSL/TLS support enables you to host E-commerce sites and accept credit card data with the highest level of security available in today's industry standards. Declaring, self-signing, and reque...
- Expected: `commercial_product/software/programming/documentation`
- Review note: web server product feature documentation / marketing
- Review confidence: 0.8
- rule prediction: `unknown/null/null/null`
- lexical prediction: `boilerplate_or_noise/web/boilerplate_or_navigation/page_noise`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: pseudo_gold_seems_ok

#### FineWeb-Edu_000003_003 (FineWeb-Edu)

- Text preview: certainly brings into focus the social aspects of disasters. The disaster trap theory, likewise, brings into focus the financial bottom line. This perspective is most often discussed in international development and disaster reduction circles. It argues that disasters destroy development gains and cause communities to de-develop unless both disaster reduction and development occur in tandem. Building a cheaper, non-earthquake resistant school in an earthquake zone, may make short-term financial ...
- Expected: `educational/science/environmental_science/article`
- Review note: disaster/ecology/social impacts discussion
- Review confidence: 0.8
- rule prediction: `educational/education/general_education/article`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `educational/education/general_education/article`
- Judgement: pseudo_gold_seems_ok

### D. Largest taxonomy gap groups, excluding previous sections

#### FineWeb_000009_013 (FineWeb)

- Text preview: the data regarding revenue-sharing cassettes from all of the other data it receives. Then, it computes the shares of the revenues due to the cassette owners or distributors and to the retailers. Reports and funds are sent to the respective participants in the revenue-sharing program according to a formula previously agreed upon. If desired, reports can be transmitted through a modem 30 and telephone lines to computers such as S1 -S4 at the places of business of the cassette owners. Alternatively...
- Expected: `reference/software/programming/documentation`
- Review note: technical business system patent description; missing_taxonomy: patents/POS_systems
- Review confidence: 0.7
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb_000009_028 (FineWeb)

- Text preview: in which the program is operating. If, however, the landmarks have not changed, as indicated by a No at step S131, processing proceeds to step S134. At step S134, the rental information which has been trapped is saved, updated during subsequent interrupt cycles, and ultimately sent to the output buffer at the end of the display. Thereafter, processing returns to step S100 to wait for the next interrupt cycle. If the determination in step S130 is negative, processing proceeds to step S140. At ste...
- Expected: `reference/software/programming/documentation`
- Review note: patent procedure for rental/return data capture; missing_taxonomy: patents/POS_systems
- Review confidence: 0.7
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb_000009_003 (FineWeb)

- Text preview: a plurality of remote video record marketing locations and for transmitting selected portions of said data to a revenue-sharing computer at a computing location, said system comprising, in combination, a point of sale system at each of a plurality of marketing locations, each of said point of sale systems comprising, in combination, data entry means for entering data concerning the sale and/or rental of merchandise, means for developing display signals corresponding to said data, means responsiv...
- Expected: `reference/software/programming/documentation`
- Review note: patent claims for data collection system; missing_taxonomy: patents/POS_systems
- Review confidence: 0.7
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb_000009_017 (FineWeb)

- Text preview: and size so that the store owner's investment is kept to a minimum. In particular, the system in Store No. 3 includes a master PC 25, a LAN adapter 27, a printer 16 and cash drawers 15 and 17. Also included are optional credit card readers 28 and 29, as well as bar code reader wands 18 and 19. In accordance with a further aspect of the present invention, the POS system also includes two simple, low-cost data input "Small Footprint Terminals" ("SFT's") 23 and 24. The small footprint terminals wil...
- Expected: `reference/software/programming/documentation`
- Review note: technical POS system patent description; missing_taxonomy: patents/POS_systems
- Review confidence: 0.7
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb_000009_052 (FineWeb)

- Text preview: for late return of the rental cassette. The Total key 530 and the Amount Tendered key 532 can be used in the manner described above to complete the transaction, and the Print button can be used to print an invoice for the further fees. If a video record is being sold, rather than rented, the operator enters the cassette identification number in response to Display 1, and presses the Sell key 516 this identifies the transaction as a sale rather than a rental. Preferably, the data regarding this t...
- Expected: `reference/software/programming/documentation`
- Review note: technical POS rental/sales transaction description; missing_taxonomy: patents/POS_systems
- Review confidence: 0.7
- rule prediction: `commercial_product/commercial/product_page/retail`
- lexical prediction: `commercial_product/commercial/product_page/retail`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `commercial_product/commercial/product_page/retail`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb_000009_042 (FineWeb)

- Text preview: to cover the "1" following the "INVOICE NUMBER" term on the top line of the screen. This area then is reduced in size, by using the "SHIFT" and the arrow keys, as it has been described above, so as to cover only the area reserved for the invoice numbers. Similarly, the other data capture areas are moved and re-shaped to cover the customer number, the type or RSCF column, the quantity or days out column, the volume identification No. column (i.e., the column entitled "NO."), and the title and the...
- Expected: `reference/software/programming/documentation`
- Review note: technical POS data capture instructions; missing_taxonomy: patents/POS_systems
- Review confidence: 0.7
- rule prediction: `educational/education/general_education/article`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `educational/education/general_education/article`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb_000009_016 (FineWeb)

- Text preview: LAN adapter 21 to the master computer 25. One of the data taps, for example, the data tap connected to terminal T1 is also connected to the printer 16. The data tap 22 is adapted to receive and temporarily store all data displayed on the display screen of the terminals T1, T2, and data to be printed on the printer 16. The captured data is thereafter supplied through the LAN adapter to the master computer. A "TSR" program disc 52 is shown in FIG. 3. As it will be described in greater detail below...
- Expected: `reference/software/programming/documentation`
- Review note: technical POS/LAN patent description; missing_taxonomy: patents/POS_systems
- Review confidence: 0.7
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb_000009_011 (FineWeb)

- Text preview: data capture procedures can be made relatively quickly and inexpensively. A novel small-footprint data entry terminal is provided to reduce the cost and size of the POS equipment used by the retailer. The small-footprint terminal has a keyboard with substantially less than a full complement of alphabetic character entry keys. Preferably, the keyboard has no such keys; rather, it has numerical keys and dedicated but programmable keys for entering specific data relating to video record transaction...
- Expected: `reference/software/programming/documentation`
- Review note: technical POS terminal patent description; missing_taxonomy: patents/POS_systems
- Review confidence: 0.7
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb_000009_038 (FineWeb)

- Text preview: type of the transaction. That is, "R" indicates a rental, "S" indicates a sale, "C" indicates a credit and "F" indicates a free transaction, such as for a free replacement. The "Day O" column indicates the number of days which this movie is rented for, which, in this example is one day. The "Day C" column indicates the number of days for which the customer is to be charged for this rental which in this situation is one day. The "Charge" column indicates the cost of the rental. The "Due Date", "D...
- Expected: `reference/software/programming/documentation`
- Review note: technical POS transaction screen description; missing_taxonomy: patents/POS_systems
- Review confidence: 0.7
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

#### FineWeb_000009_002 (FineWeb)

- Text preview: revenue-sharing system for computing and sorting shares of rental revenues, said master computer of said local area network being programmed with software operable with that of the computer at said remote location, said data collection means being connected to said data entry means and operating to deliver said predetermined data to said remote location in compatible form, without altering the programming of said master computer. 12. A data collection device for selecting predetermined output da...
- Expected: `reference/software/programming/documentation`
- Review note: patent claim for revenue-sharing system; missing_taxonomy: patents/POS_systems
- Review confidence: 0.7
- rule prediction: `unknown/null/null/null`
- lexical prediction: `unknown/null/null/null`
- MiniLM prediction: `unknown/null/null/null`
- hybrid prediction: `unknown/null/null/null`
- Judgement: taxonomy_gap; pseudo_gold_seems_ok

## Reliability assessment

- Clearly questionable pseudo-gold labels found by this audit: 3 low-confidence records in the full file; 3 appear in the 40-record spot-check.
- A much larger set is taxonomy-gap-limited rather than obviously wrong: labels are reasonable nearest matches, but the taxonomy lacks the exact category.
- Current metrics can be treated as preliminary pipeline/debugging metrics.
- Current metrics should not be treated as final model-quality metrics until a human reviews pseudo-gold labels and taxonomy coverage is explicitly frozen or expanded.

## Recommendations before tuning

1. Human-review the low-confidence and taxonomy-gap records first; do not tune on potentially noisy pseudo-gold labels.
2. Add duplicate `chunk_id` detection to the evaluator before using it as a reusable benchmark tool.
3. Decide whether the MVP taxonomy should expand to cover patents, statistics/probability, chemistry, arithmetic/measurement, and hardware/networking.
4. Only after the label set is reviewed, rerun evaluation and then consider MiniLM threshold/top-k/taxonomy-description tuning.
5. Keep the current metrics framed as a baseline failure/readiness signal, not a final classifier result.
