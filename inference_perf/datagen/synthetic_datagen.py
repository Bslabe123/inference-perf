# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from inference_perf.apis import InferenceAPIData, CompletionAPIData
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.utils.distribution import generate_distribution
from .base import DataGenerator
from typing import Generator, List
from inference_perf.config import APIConfig, APIType, DataConfig


class SyntheticDataGenerator(DataGenerator):
    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: CustomTokenizer) -> None:
        super().__init__(api_config, config, tokenizer)

        if self.input_distribution is None or self.output_distribution is None or self.tokenizer is None:
            raise ValueError("IODistribution and tokenizer are required for SyntheticDataGenerator")

        if self.input_distribution.total_count is None or self.output_distribution.total_count is None:
            raise ValueError("IODistribution requires total_count to be set")

        self.input_lengths = generate_distribution(
            self.input_distribution.min,
            self.input_distribution.max,
            self.input_distribution.mean,
            self.input_distribution.std_dev,
            self.input_distribution.total_count,
        )
        self.output_lengths = generate_distribution(
            self.output_distribution.min,
            self.output_distribution.max,
            self.output_distribution.mean,
            self.output_distribution.std_dev,
            self.output_distribution.total_count,
        )
        base_prompt = "Pick as many lines as you can from these poem lines:\n"
        self.token_ids = self.tokenizer.get_tokenizer().encode(base_prompt + self.get_sonnet_data())

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return False

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        i = 0
        while True:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for SyntheticDataGenerator")
            if self.api_config.type == APIType.Completion:
                yield CompletionAPIData(
                    prompt=self.tokenizer.get_tokenizer().decode(self.token_ids[: self.input_lengths[i]]),
                    max_tokens=self.output_lengths[i],
                )
                i += 1
            else:
                raise Exception("Unsupported API type")

    # Hardcoded sonnet data that we can use for synthetic benchmarks.
    def get_sonnet_data(self) -> str:
        return """FROM fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:
But thou, contracted to thine own bright eyes,
Feed'st thy light'st flame with self-substantial fuel,
Making a famine where abundance lies,
Thyself thy foe, to thy sweet self too cruel.
Thou that art now the world's fresh ornament
And only herald to the gaudy spring,
Within thine own bud buriest thy content
And, tender churl, makest waste in niggarding.
Pity the world, or else this glutton be,
To eat the world's due, by the grave and thee.
When forty winters shall beseige thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery, so gazed on now,
Will be a tatter'd weed, of small worth held:
Then being ask'd where all thy beauty lies,
Where all the treasure of thy lusty days,
To say, within thine own deep-sunken eyes,
Were an all-eating shame and thriftless praise.
How much more praise deserved thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.
Look in thy glass, and tell the face thou viewest
Now is the time that face should form another;
Whose fresh repair if now thou not renewest,
Thou dost beguile the world, unbless some mother.
For where is she so fair whose unear'd womb
Disdains the tillage of thy husbandry?
Or who is he so fond will be the tomb
Of his self-love, to stop posterity?
Thou art thy mother's glass, and she in thee
Calls back the lovely April of her prime:
So thou through windows of thine age shall see
Despite of wrinkles this thy golden time.
But if thou live, remember'd not to be,
Die single, and thine image dies with thee.
Unthrifty loveliness, why dost thou spend
Upon thyself thy beauty's legacy?
Nature's bequest gives nothing but doth lend,
And being frank she lends to those are free.
Then, beauteous niggard, why dost thou abuse
The bounteous largess given thee to give?
Profitless usurer, why dost thou use
So great a sum of sums, yet canst not live?
For having traffic with thyself alone,
Thou of thyself thy sweet self dost deceive.
Then how, when nature calls thee to be gone,
What acceptable audit canst thou leave?
Thy unused beauty must be tomb'd with thee,
Which, used, lives th' executor to be.
Those hours, that with gentle work did frame
The lovely gaze where every eye doth dwell,
Will play the tyrants to the very same
And that unfair which fairly doth excel:
For never-resting time leads summer on
To hideous winter and confounds him there;
Sap cheque'd with frost and lusty leaves quite gone,
Beauty o'ersnow'd and bareness every where:
Then, were not summer's distillation left,
A liquid prisoner pent in walls of glass,
Beauty's effect with beauty were bereft,
Nor it nor no remembrance what it was:
But flowers distill'd though they with winter meet,
Leese but their show; their substance still lives sweet.
Then let not winter's ragged hand deface
In thee thy summer, ere thou be distill'd:
Make sweet some vial; treasure thou some place
With beauty's treasure, ere it be self-kill'd.
That use is not forbidden usury,
Which happies those that pay the willing loan;
That's for thyself to breed another thee,
Or ten times happier, be it ten for one;
Ten times thyself were happier than thou art,
If ten of thine ten times refigured thee:
Then what could death do, if thou shouldst depart,
Leaving thee living in posterity?
Be not self-will'd, for thou art much too fair
To be death's conquest and make worms thine heir.
Lo! in the orient when the gracious light
Lifts up his burning head, each under eye
Doth homage to his new-appearing sight,
Serving with looks his sacred majesty;
And having climb'd the steep-up heavenly hill,
Resembling strong youth in his middle age,
yet mortal looks adore his beauty still,
Attending on his golden pilgrimage;
But when from highmost pitch, with weary car,
Like feeble age, he reeleth from the day,
The eyes, 'fore duteous, now converted are
From his low tract and look another way:
So thou, thyself out-going in thy noon,
Unlook'd on diest, unless thou get a son.
Music to hear, why hear'st thou music sadly?
Sweets with sweets war not, joy delights in joy.
Why lovest thou that which thou receivest not gladly,
Or else receivest with pleasure thine annoy?
If the true concord of well-tuned sounds,
By unions married, do offend thine ear,
They do but sweetly chide thee, who confounds
In singleness the parts that thou shouldst bear.
Mark how one string, sweet husband to another,
Strikes each in each by mutual ordering,
Resembling sire and child and happy mother
Who all in one, one pleasing note do sing:
Whose speechless song, being many, seeming one,
Sings this to thee: 'thou single wilt prove none.'
Is it for fear to wet a widow's eye
That thou consumest thyself in single life?
Ah! if thou issueless shalt hap to die.
The world will wail thee, like a makeless wife;
The world will be thy widow and still weep
That thou no form of thee hast left behind,
When every private widow well may keep
By children's eyes her husband's shape in mind.
Look, what an unthrift in the world doth spend
Shifts but his place, for still the world enjoys it;
But beauty's waste hath in the world an end,
And kept unused, the user so destroys it.
No love toward others in that bosom sits
That on himself such murderous shame commits.
For shame! deny that thou bear'st love to any,
Who for thyself art so unprovident.
Grant, if thou wilt, thou art beloved of many,
But that thou none lovest is most evident;
For thou art so possess'd with murderous hate
That 'gainst thyself thou stick'st not to conspire.
Seeking that beauteous roof to ruinate
Which to repair should be thy chief desire.
O, change thy thought, that I may change my mind!
Shall hate be fairer lodged than gentle love?
Be, as thy presence is, gracious and kind,
Or to thyself at least kind-hearted prove:
Make thee another self, for love of me,
That beauty still may live in thine or thee.
As fast as thou shalt wane, so fast thou growest
In one of thine, from that which thou departest;
And that fresh blood which youngly thou bestowest
Thou mayst call thine when thou from youth convertest.
Herein lives wisdom, beauty and increase:
Without this, folly, age and cold decay:
If all were minded so, the times should cease
And threescore year would make the world away.
Let those whom Nature hath not made for store,
Harsh featureless and rude, barrenly perish:
Look, whom she best endow'd she gave the more;
Which bounteous gift thou shouldst in bounty cherish:
She carved thee for her seal, and meant thereby
Thou shouldst print more, not let that copy die.
When I do count the clock that tells the time,
And see the brave day sunk in hideous night;
When I behold the violet past prime,
And sable curls all silver'd o'er with white;
When lofty trees I see barren of leaves
Which erst from heat did canopy the herd,
And summer's green all girded up in sheaves
Borne on the bier with white and bristly beard,
Then of thy beauty do I question make,
That thou among the wastes of time must go,
Since sweets and beauties do themselves forsake
And die as fast as they see others grow;
And nothing 'gainst Time's scythe can make defence
Save breed, to brave him when he takes thee hence.
O, that you were yourself! but, love, you are
No longer yours than you yourself here live:
Against this coming end you should prepare,
And your sweet semblance to some other give.
So should that beauty which you hold in lease
Find no determination: then you were
Yourself again after yourself's decease,
When your sweet issue your sweet form should bear.
Who lets so fair a house fall to decay,
Which husbandry in honour might uphold
Against the stormy gusts of winter's day
And barren rage of death's eternal cold?
O, none but unthrifts! Dear my love, you know
You had a father: let your son say so.
Not from the stars do I my judgment pluck;
And yet methinks I have astronomy,
But not to tell of good or evil luck,
Of plagues, of dearths, or seasons' quality;
Nor can I fortune to brief minutes tell,
Pointing to each his thunder, rain and wind,
Or say with princes if it shall go well,
By oft predict that I in heaven find:
But from thine eyes my knowledge I derive,
And, constant stars, in them I read such art
As truth and beauty shall together thrive,
If from thyself to store thou wouldst convert;
Or else of thee this I prognosticate:
Thy end is truth's and beauty's doom and date.
When I consider every thing that grows
Holds in perfection but a little moment,
That this huge stage presenteth nought but shows
Whereon the stars in secret influence comment;
When I perceive that men as plants increase,
Cheered and cheque'd even by the self-same sky,
Vaunt in their youthful sap, at height decrease,
And wear their brave state out of memory;
Then the conceit of this inconstant stay
Sets you most rich in youth before my sight,
Where wasteful Time debateth with Decay,
To change your day of youth to sullied night;
And all in war with Time for love of you,
As he takes from you, I engraft you new.
But wherefore do not you a mightier way
Make war upon this bloody tyrant, Time?
And fortify yourself in your decay
With means more blessed than my barren rhyme?
Now stand you on the top of happy hours,
And many maiden gardens yet unset
With virtuous wish would bear your living flowers,
Much liker than your painted counterfeit:
So should the lines of life that life repair,
Which this, Time's pencil, or my pupil pen,
Neither in inward worth nor outward fair,
Can make you live yourself in eyes of men.
To give away yourself keeps yourself still,
And you must live, drawn by your own sweet skill.
Who will believe my verse in time to come,
If it were fill'd with your most high deserts?
Though yet, heaven knows, it is but as a tomb
Which hides your life and shows not half your parts.
If I could write the beauty of your eyes
And in fresh numbers number all your graces,
The age to come would say 'This poet lies:
Such heavenly touches ne'er touch'd earthly faces.'
So should my papers yellow'd with their age
Be scorn'd like old men of less truth than tongue,
And your true rights be term'd a poet's rage
And stretched metre of an antique song:
But were some child of yours alive that time,
You should live twice; in it and in my rhyme.
Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade
Nor lose possession of that fair thou owest;
Nor shall Death brag thou wander'st in his shade,
When in eternal lines to time thou growest:
So long as men can breathe or eyes can see,
So long lives this and this gives life to thee.
Devouring Time, blunt thou the lion's paws,
And make the earth devour her own sweet brood;
Pluck the keen teeth from the fierce tiger's jaws,
And burn the long-lived phoenix in her blood;
Make glad and sorry seasons as thou fleets,
And do whate'er thou wilt, swift-footed Time,
To the wide world and all her fading sweets;
But I forbid thee one most heinous crime:
O, carve not with thy hours my love's fair brow,
Nor draw no lines there with thine antique pen;
Him in thy course untainted do allow
For beauty's pattern to succeeding men.
Yet, do thy worst, old Time: despite thy wrong,
My love shall in my verse ever live young.
A woman's face with Nature's own hand painted
Hast thou, the master-mistress of my passion;
A woman's gentle heart, but not acquainted
With shifting change, as is false women's fashion;
An eye more bright than theirs, less false in rolling,
Gilding the object whereupon it gazeth;
A man in hue, all 'hues' in his controlling,
Much steals men's eyes and women's souls amazeth.
And for a woman wert thou first created;
Till Nature, as she wrought thee, fell a-doting,
And by addition me of thee defeated,
By adding one thing to my purpose nothing.
But since she prick'd thee out for women's pleasure,
Mine be thy love and thy love's use their treasure.
So is it not with me as with that Muse
Stirr'd by a painted beauty to his verse,
Who heaven itself for ornament doth use
And every fair with his fair doth rehearse
Making a couplement of proud compare,
With sun and moon, with earth and sea's rich gems,
With April's first-born flowers, and all things rare
That heaven's air in this huge rondure hems.
O' let me, true in love, but truly write,
And then believe me, my love is as fair
As any mother's child, though not so bright
As those gold candles fix'd in heaven's air:
Let them say more than like of hearsay well;
I will not praise that purpose not to sell.
My glass shall not persuade me I am old,
So long as youth and thou are of one date;
But when in thee time's furrows I behold,
Then look I death my days should expiate.
For all that beauty that doth cover thee
Is but the seemly raiment of my heart,
Which in thy breast doth live, as thine in me:
How can I then be elder than thou art?
O, therefore, love, be of thyself so wary
As I, not for myself, but for thee will;
Bearing thy heart, which I will keep so chary
As tender nurse her babe from faring ill.
Presume not on thy heart when mine is slain;
Thou gavest me thine, not to give back again.
As an unperfect actor on the stage
Who with his fear is put besides his part,
Or some fierce thing replete with too much rage,
Whose strength's abundance weakens his own heart.
So I, for fear of trust, forget to say
The perfect ceremony of love's rite,
And in mine own love's strength seem to decay,
O'ercharged with burden of mine own love's might.
O, let my books be then the eloquence
And dumb presagers of my speaking breast,
Who plead for love and look for recompense
More than that tongue that more hath more express'd.
O, learn to read what silent love hath writ:
To hear with eyes belongs to love's fine wit.
Mine eye hath play'd the painter and hath stell'd
Thy beauty's form in table of my heart;
My body is the frame wherein 'tis held,
And perspective it is the painter's art.
For through the painter must you see his skill,
To find where your true image pictured lies;
Which in my bosom's shop is hanging still,
That hath his windows glazed with thine eyes.
Now see what good turns eyes for eyes have done:
Mine eyes have drawn thy shape, and thine for me
Are windows to my breast, where-through the sun
Delights to peep, to gaze therein on thee;
Yet eyes this cunning want to grace their art;
They draw but what they see, know not the heart.
Let those who are in favour with their stars
Of public honour and proud titles boast,
Whilst I, whom fortune of such triumph bars,
Unlook'd for joy in that I honour most.
Great princes' favourites their fair leaves spread
But as the marigold at the sun's eye,
And in themselves their pride lies buried,
For at a frown they in their glory die.
The painful warrior famoused for fight,
After a thousand victories once foil'd,
Is from the book of honour razed quite,
And all the rest forgot for which he toil'd:
Then happy I, that love and am beloved
Where I may not remove nor be removed.
Lord of my love, to whom in vassalage
Thy merit hath my duty strongly knit,
To thee I send this written embassage,
To witness duty, not to show my wit:
Duty so great, which wit so poor as mine
May make seem bare, in wanting words to show it,
But that I hope some good conceit of thine
In thy soul's thought, all naked, will bestow it;
Till whatsoever star that guides my moving
Points on me graciously with fair aspect
And puts apparel on my tatter'd loving,
To show me worthy of thy sweet respect:
Then may I dare to boast how I do love thee;
Till then not show my head where thou mayst prove me.
Weary with toil, I haste me to my bed,
The dear repose for limbs with travel tired;
But then begins a journey in my head,
To work my mind, when body's work's expired:
For then my thoughts, from far where I abide,
Intend a zealous pilgrimage to thee,
And keep my drooping eyelids open wide,
Looking on darkness which the blind do see
Save that my soul's imaginary sight
Presents thy shadow to my sightless view,
Which, like a jewel hung in ghastly night,
Makes black night beauteous and her old face new.
Lo! thus, by day my limbs, by night my mind,
For thee and for myself no quiet find.
How can I then return in happy plight,
That am debarr'd the benefit of rest?
When day's oppression is not eased by night,
But day by night, and night by day, oppress'd?
And each, though enemies to either's reign,
Do in consent shake hands to torture me;
The one by toil, the other to complain
How far I toil, still farther off from thee.
I tell the day, to please them thou art bright
And dost him grace when clouds do blot the heaven:
So flatter I the swart-complexion'd night,
When sparkling stars twire not thou gild'st the even.
But day doth daily draw my sorrows longer
And night doth nightly make grief's strength seem stronger.
When, in disgrace with fortune and men's eyes,
I all alone beweep my outcast state
And trouble deal heaven with my bootless cries
And look upon myself and curse my fate,
Wishing me like to one more rich in hope,
Featured like him, like him with friends possess'd,
Desiring this man's art and that man's scope,
With what I most enjoy contented least;
Yet in these thoughts myself almost despising,
Haply I think on thee, and then my state,
Like to the lark at break of day arising
From sullen earth, sings hymns at heaven's gate;
For thy sweet love remember'd such wealth brings
That then I scorn to change my state with kings.
When to the sessions of sweet silent thought
I summon up remembrance of things past,
I sigh the lack of many a thing I sought,
And with old woes new wail my dear time's waste:
Then can I drown an eye, unused to flow,
For precious friends hid in death's dateless night,
And weep afresh love's long since cancell'd woe,
And moan the expense of many a vanish'd sight:
Then can I grieve at grievances foregone,
And heavily from woe to woe tell o'er
The sad account of fore-bemoaned moan,
Which I new pay as if not paid before.
But if the while I think on thee, dear friend,
All losses are restored and sorrows end.
Thy bosom is endeared with all hearts,
Which I by lacking have supposed dead,
And there reigns love and all love's loving parts,
And all those friends which I thought buried.
How many a holy and obsequious tear
Hath dear religious love stol'n from mine eye
As interest of the dead, which now appear
But things removed that hidden in thee lie!
Thou art the grave where buried love doth live,
Hung with the trophies of my lovers gone,
Who all their parts of me to thee did give;
That due of many now is thine alone:
Their images I loved I view in thee,
And thou, all they, hast all the all of me.
If thou survive my well-contented day,
When that churl Death my bones with dust shall cover,
And shalt by fortune once more re-survey
These poor rude lines of thy deceased lover,
Compare them with the bettering of the time,
And though they be outstripp'd by every pen,
Reserve them for my love, not for their rhyme,
Exceeded by the height of happier men.
O, then vouchsafe me but this loving thought:
'Had my friend's Muse grown with this growing age,
A dearer birth than this his love had brought,
To march in ranks of better equipage:
But since he died and poets better prove,
Theirs for their style I'll read, his for his love.'
Full many a glorious morning have I seen
Flatter the mountain-tops with sovereign eye,
Kissing with golden face the meadows green,
Gilding pale streams with heavenly alchemy;
Anon permit the basest clouds to ride
With ugly rack on his celestial face,
And from the forlorn world his visage hide,
Stealing unseen to west with this disgrace:
Even so my sun one early morn did shine
With all triumphant splendor on my brow;
But out, alack! he was but one hour mine;
The region cloud hath mask'd him from me now.
Yet him for this my love no whit disdaineth;
Suns of the world may stain when heaven's sun staineth.
Why didst thou promise such a beauteous day,
And make me travel forth without my cloak,
To let base clouds o'ertake me in my way,
Hiding thy bravery in their rotten smoke?
'Tis not enough that through the cloud thou break,
To dry the rain on my storm-beaten face,
For no man well of such a salve can speak
That heals the wound and cures not the disgrace:
Nor can thy shame give physic to my grief;
Though thou repent, yet I have still the loss:
The offender's sorrow lends but weak relief
To him that bears the strong offence's cross.
Ah! but those tears are pearl which thy love sheds,
And they are rich and ransom all ill deeds.
No more be grieved at that which thou hast done:
Roses have thorns, and silver fountains mud;
Clouds and eclipses stain both moon and sun,
And loathsome canker lives in sweetest bud.
All men make faults, and even I in this,
Authorizing thy trespass with compare,
Myself corrupting, salving thy amiss,
Excusing thy sins more than thy sins are;
For to thy sensual fault I bring in sense--
Thy adverse party is thy advocate--
And 'gainst myself a lawful plea commence:
Such civil war is in my love and hate
That I an accessary needs must be
To that sweet thief which sourly robs from me.
Let me confess that we two must be twain,
Although our undivided loves are one:
So shall those blots that do with me remain
Without thy help by me be borne alone.
In our two loves there is but one respect,
Though in our lives a separable spite,
Which though it alter not love's sole effect,
Yet doth it steal sweet hours from love's delight.
I may not evermore acknowledge thee,
Lest my bewailed guilt should do thee shame,
Nor thou with public kindness honour me,
Unless thou take that honour from thy name:
But do not so; I love thee in such sort
As, thou being mine, mine is thy good report.
As a decrepit father takes delight
To see his active child do deeds of youth,
So I, made lame by fortune's dearest spite,
Take all my comfort of thy worth and truth.
For whether beauty, birth, or wealth, or wit,
Or any of these all, or all, or more,
Entitled in thy parts do crowned sit,
I make my love engrafted to this store:
So then I am not lame, poor, nor despised,
Whilst that this shadow doth such substance give
That I in thy abundance am sufficed
And by a part of all thy glory live.
Look, what is best, that best I wish in thee:
This wish I have; then ten times happy me!"""
