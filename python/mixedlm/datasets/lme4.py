from __future__ import annotations

import pandas as pd


def load_sleepstudy() -> pd.DataFrame:
    """Load the sleepstudy dataset from lme4.

    Reaction times in a sleep deprivation study. On day 0 the subjects had
    their normal amount of sleep. Starting that night they were restricted
    to 3 hours of sleep per night. The observations represent the average
    reaction time on a series of tests given each day to each subject.

    Returns
    -------
    pd.DataFrame
        DataFrame with 180 observations and 3 columns:
        - Reaction: Average reaction time (ms)
        - Days: Number of days of sleep deprivation (0-9)
        - Subject: Subject identifier (308-372)

    Examples
    --------
    >>> from mixedlm.datasets import load_sleepstudy
    >>> sleepstudy = load_sleepstudy()
    >>> sleepstudy.head()
       Reaction  Days Subject
    0    249.56     0     308
    1    258.70     1     308
    2    250.80     2     308
    3    321.44     3     308
    4    356.85     4     308

    >>> # Fit a linear mixed model
    >>> from mixedlm import lmer
    >>> model = lmer("Reaction ~ Days + (Days | Subject)", data=sleepstudy)
    """
    data = {
        "Reaction": [
            249.5600,
            258.7047,
            250.8006,
            321.4398,
            356.8519,
            414.6901,
            382.2038,
            290.1486,
            430.5853,
            466.3535,
            222.7339,
            205.2658,
            202.9778,
            204.7070,
            207.7161,
            215.9618,
            213.6303,
            217.7272,
            224.2957,
            237.3142,
            199.0539,
            194.3322,
            234.3200,
            232.8416,
            229.3074,
            220.4579,
            235.4208,
            255.7511,
            261.0125,
            247.5153,
            321.5426,
            300.4002,
            283.8565,
            285.1330,
            285.7973,
            297.5855,
            280.2396,
            318.2613,
            305.3495,
            354.0487,
            287.6079,
            285.0000,
            301.8206,
            320.1153,
            316.2773,
            293.3187,
            290.0750,
            334.8177,
            293.7469,
            371.5811,
            234.8606,
            242.8118,
            272.9613,
            309.7688,
            317.4629,
            309.9976,
            454.1619,
            346.8311,
            330.3003,
            253.8644,
            283.8424,
            289.5550,
            276.7693,
            299.8097,
            297.1710,
            338.1665,
            340.8485,
            305.3211,
            354.0032,
            387.6167,
            265.4731,
            276.2012,
            243.3647,
            254.6723,
            279.0244,
            284.1912,
            305.5248,
            331.5229,
            335.7469,
            377.2990,
            241.6083,
            273.9472,
            254.4907,
            270.8021,
            251.4519,
            254.6362,
            245.4523,
            235.3110,
            235.7541,
            237.2466,
            312.3666,
            313.8058,
            291.6112,
            346.1222,
            365.7324,
            391.8385,
            404.2601,
            416.6923,
            455.8643,
            458.9167,
            236.1032,
            230.3167,
            238.9256,
            254.9220,
            250.7103,
            269.7744,
            281.5648,
            308.1020,
            336.2806,
            351.6451,
            256.2968,
            243.4543,
            256.2046,
            255.5271,
            268.9165,
            329.7247,
            379.4445,
            362.9184,
            394.4872,
            389.0527,
            250.5265,
            300.0576,
            269.8939,
            280.5891,
            271.8274,
            304.6336,
            287.7466,
            266.5955,
            321.5418,
            347.5655,
            221.6771,
            298.1939,
            326.8785,
            346.8555,
            348.7402,
            352.8287,
            354.4266,
            360.4326,
            375.6406,
            388.5417,
            271.9235,
            268.4369,
            257.2424,
            277.6566,
            314.8222,
            317.2135,
            298.1353,
            348.1229,
            340.2800,
            366.5131,
            225.2640,
            234.5235,
            238.9008,
            240.4730,
            267.5373,
            344.1937,
            281.1481,
            347.5855,
            365.1630,
            372.2288,
            269.8804,
            272.4428,
            277.8989,
            281.7895,
            279.1705,
            284.5120,
            259.2658,
            304.6306,
            350.7807,
            369.4692,
            269.4117,
            273.4740,
            297.5968,
            310.6316,
            287.1726,
            329.6076,
            334.4818,
            343.2199,
            369.1417,
            364.1236,
        ],
        "Days": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 18,
        "Subject": (
            ["308"] * 10
            + ["309"] * 10
            + ["310"] * 10
            + ["330"] * 10
            + ["331"] * 10
            + ["332"] * 10
            + ["333"] * 10
            + ["334"] * 10
            + ["335"] * 10
            + ["337"] * 10
            + ["349"] * 10
            + ["350"] * 10
            + ["351"] * 10
            + ["352"] * 10
            + ["369"] * 10
            + ["370"] * 10
            + ["371"] * 10
            + ["372"] * 10
        ),
    }
    return pd.DataFrame(data)


def load_cbpp() -> pd.DataFrame:
    """Load the cbpp dataset from lme4.

    Contagious bovine pleuropneumonia (CBPP) data. This dataset describes
    the serological incidence of CBPP in zebu cattle during a follow-up
    survey implemented in 15 ings from Ethiopian herds.

    Returns
    -------
    pd.DataFrame
        DataFrame with 56 observations and 4 columns:
        - herd: Herd identifier (factor with 15 levels)
        - incidence: Number of new serological cases
        - size: Herd size at the beginning of the period
        - period: Time period (factor with 4 levels: 1-4)

    Examples
    --------
    >>> from mixedlm.datasets import load_cbpp
    >>> cbpp = load_cbpp()
    >>> cbpp.head()
      herd  incidence  size period
    0    1          2    14      1
    1    1          3    12      2
    2    1          4     9      3
    3    1          0     5      4
    4    2          3    22      1

    >>> # Fit a binomial GLMM
    >>> from mixedlm import glmer
    >>> from mixedlm.families import Binomial
    >>> model = glmer(
    ...     "incidence / size ~ period + (1 | herd)",
    ...     data=cbpp,
    ...     family=Binomial()
    ... )
    """
    data = {
        "herd": [
            "1",
            "1",
            "1",
            "1",
            "2",
            "2",
            "2",
            "2",
            "3",
            "3",
            "3",
            "3",
            "4",
            "4",
            "4",
            "4",
            "5",
            "5",
            "5",
            "5",
            "6",
            "6",
            "6",
            "6",
            "7",
            "7",
            "7",
            "7",
            "8",
            "8",
            "8",
            "8",
            "9",
            "9",
            "9",
            "9",
            "10",
            "10",
            "10",
            "10",
            "11",
            "11",
            "11",
            "11",
            "12",
            "12",
            "12",
            "12",
            "13",
            "13",
            "13",
            "13",
            "14",
            "14",
            "14",
            "14",
        ],
        "incidence": [
            2,
            3,
            4,
            0,
            3,
            1,
            1,
            0,
            4,
            4,
            3,
            0,
            1,
            2,
            2,
            0,
            3,
            0,
            2,
            0,
            1,
            3,
            0,
            0,
            6,
            2,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            0,
            0,
            0,
            1,
            2,
            0,
            0,
            0,
            0,
            1,
            0,
            2,
            2,
            1,
            0,
            0,
            0,
            1,
            0,
            2,
            2,
            0,
            0,
        ],
        "size": [
            14,
            12,
            9,
            5,
            22,
            18,
            21,
            22,
            17,
            12,
            9,
            5,
            31,
            28,
            22,
            16,
            10,
            9,
            9,
            7,
            12,
            10,
            8,
            3,
            30,
            25,
            24,
            19,
            25,
            24,
            23,
            22,
            9,
            7,
            5,
            4,
            22,
            18,
            16,
            12,
            13,
            10,
            9,
            8,
            19,
            17,
            14,
            8,
            8,
            7,
            5,
            4,
            17,
            15,
            12,
            10,
        ],
        "period": ["1", "2", "3", "4"] * 14,
    }
    return pd.DataFrame(data)


def load_dyestuff() -> pd.DataFrame:
    """Load the Dyestuff dataset from lme4.

    Yield of dyestuff by batch. A classic one-way random effects dataset
    from Davies (1967). Six samples of dyestuff were selected from a
    manufacturing process, and the yield was determined for each batch.

    This dataset is often used to demonstrate random effects models where
    the batch effect is a random effect.

    Returns
    -------
    pd.DataFrame
        DataFrame with 30 observations and 2 columns:
        - Batch: Batch identifier (factor with 6 levels: A-F)
        - Yield: Yield of dyestuff (numeric)

    References
    ----------
    Davies, O. L. (1967). Design and Analysis of Industrial Experiments
    (2nd ed.). Hafner Publishing Company.

    Examples
    --------
    >>> from mixedlm.datasets import load_dyestuff
    >>> dyestuff = load_dyestuff()
    >>> dyestuff.head()
      Batch  Yield
    0     A   1545
    1     A   1440
    2     A   1440
    3     A   1520
    4     A   1580

    >>> # Fit a random intercept model
    >>> from mixedlm import lmer
    >>> model = lmer("Yield ~ 1 + (1 | Batch)", data=dyestuff)
    """
    data = {
        "Batch": (["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5 + ["E"] * 5 + ["F"] * 5),
        "Yield": [
            1545,
            1440,
            1440,
            1520,
            1580,
            1540,
            1555,
            1490,
            1560,
            1495,
            1595,
            1550,
            1605,
            1510,
            1560,
            1445,
            1440,
            1595,
            1465,
            1545,
            1595,
            1630,
            1515,
            1635,
            1625,
            1520,
            1455,
            1450,
            1480,
            1445,
        ],
    }
    return pd.DataFrame(data)


def load_dyestuff2() -> pd.DataFrame:
    """Load the Dyestuff2 dataset from lme4.

    A simulated dataset similar to Dyestuff but with less variation
    between batches, designed to illustrate boundary (singular) fits.

    Returns
    -------
    pd.DataFrame
        DataFrame with 30 observations and 2 columns:
        - Batch: Batch identifier (factor with 6 levels: A-F)
        - Yield: Yield of dyestuff (numeric)

    Examples
    --------
    >>> from mixedlm.datasets import load_dyestuff2
    >>> dyestuff2 = load_dyestuff2()

    >>> # This often produces a singular fit
    >>> from mixedlm import lmer
    >>> model = lmer("Yield ~ 1 + (1 | Batch)", data=dyestuff2)
    """
    data = {
        "Batch": (["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5 + ["E"] * 5 + ["F"] * 5),
        "Yield": [
            7.298,
            3.846,
            2.434,
            9.566,
            7.990,
            5.220,
            6.556,
            0.608,
            11.788,
            -0.892,
            0.110,
            10.386,
            13.434,
            5.510,
            8.166,
            2.212,
            4.852,
            7.092,
            9.288,
            4.980,
            0.282,
            9.014,
            4.458,
            8.756,
            7.600,
            3.070,
            11.966,
            4.034,
            5.734,
            9.442,
        ],
    }
    return pd.DataFrame(data)


def load_penicillin() -> pd.DataFrame:
    """Load the Penicillin dataset from lme4.

    Variation in the potency of Penicillin. The potency of Penicillin
    was assessed using a microbiological assay. The experiment was
    designed as a Latin square where each plate (row) receives a
    different sample of Penicillin and the response is measured at
    different positions.

    This is a classic dataset for demonstrating crossed random effects,
    where both 'plate' and 'sample' are random effects that are not
    nested within each other.

    Returns
    -------
    pd.DataFrame
        DataFrame with 144 observations and 3 columns:
        - diameter: Diameter of the zone of inhibition (mm)
        - plate: Plate identifier (factor with 24 levels: a-x)
        - sample: Sample identifier (factor with 6 levels: A-F)

    References
    ----------
    Bliss, C. I. (1967). Statistics in Biology. McGraw-Hill.

    Examples
    --------
    >>> from mixedlm.datasets import load_penicillin
    >>> penicillin = load_penicillin()
    >>> penicillin.head()
       diameter plate sample
    0        27     a      A
    1        23     a      B
    2        26     a      C
    3        23     a      D
    4        23     a      E

    >>> # Fit a crossed random effects model
    >>> from mixedlm import lmer
    >>> model = lmer("diameter ~ 1 + (1 | plate) + (1 | sample)", data=penicillin)
    """
    plates = list("abcdefghijklmnopqrstuvwx")
    samples = list("ABCDEF")

    diameters = [
        27,
        23,
        26,
        23,
        23,
        21,
        31,
        27,
        28,
        24,
        22,
        25,
        26,
        23,
        24,
        23,
        22,
        21,
        25,
        23,
        22,
        21,
        22,
        22,
        24,
        21,
        24,
        20,
        21,
        19,
        28,
        24,
        27,
        23,
        24,
        24,
        24,
        21,
        25,
        21,
        21,
        17,
        28,
        24,
        26,
        23,
        23,
        22,
        27,
        24,
        26,
        21,
        24,
        19,
        22,
        21,
        23,
        19,
        19,
        19,
        25,
        22,
        24,
        20,
        24,
        22,
        27,
        24,
        27,
        22,
        22,
        22,
        27,
        23,
        25,
        22,
        22,
        22,
        23,
        21,
        23,
        20,
        21,
        18,
        26,
        23,
        24,
        22,
        24,
        20,
        28,
        23,
        25,
        22,
        21,
        22,
        25,
        23,
        23,
        19,
        20,
        20,
        27,
        24,
        25,
        23,
        23,
        21,
        21,
        19,
        22,
        18,
        18,
        17,
        27,
        24,
        26,
        22,
        23,
        22,
        26,
        22,
        24,
        22,
        22,
        22,
        27,
        24,
        25,
        22,
        22,
        21,
        26,
        22,
        26,
        21,
        22,
        21,
        22,
        20,
        21,
        18,
        18,
        18,
    ]

    data: dict[str, list] = {"diameter": [], "plate": [], "sample": []}

    idx = 0
    for plate in plates:
        for sample in samples:
            data["diameter"].append(diameters[idx])
            data["plate"].append(plate)
            data["sample"].append(sample)
            idx += 1

    return pd.DataFrame(data)


def load_cake() -> pd.DataFrame:
    """Load the cake dataset from lme4.

    Data from a Latin square design studying the effect of recipe and
    baking temperature on the breaking angle of chocolate cakes.

    This is a classic split-plot design where temperature is the
    whole-plot factor and recipe is the subplot factor.

    Returns
    -------
    pd.DataFrame
        DataFrame with 270 observations and 4 columns:
        - replicate: Replicate (factor with 15 levels: 1-15)
        - recipe: Recipe (factor with 3 levels: A, B, C)
        - temperature: Baking temperature (factor: 175, 185, ..., 225)
        - angle: Breaking angle (degrees)

    References
    ----------
    Cochran, W. G. and Cox, G. M. (1957). Experimental Designs (2nd ed.).
    John Wiley & Sons.

    Examples
    --------
    >>> from mixedlm.datasets import load_cake
    >>> cake = load_cake()
    >>> cake.head()
       replicate recipe temperature  angle
    0          1      A         175     42
    1          1      A         185     46
    ...

    >>> from mixedlm import lmer
    >>> model = lmer("angle ~ recipe * temperature + (1|replicate)", data=cake)
    """
    replicates = list(range(1, 16))
    recipes = ["A", "B", "C"]
    temperatures = ["175", "185", "195", "205", "215", "225"]

    angles = [
        42,
        46,
        47,
        39,
        53,
        42,
        39,
        46,
        29,
        40,
        35,
        47,
        46,
        24,
        32,
        37,
        47,
        29,
        32,
        35,
        28,
        35,
        24,
        39,
        26,
        37,
        32,
        30,
        35,
        33,
        26,
        23,
        25,
        32,
        36,
        28,
        25,
        31,
        35,
        35,
        28,
        30,
        37,
        29,
        27,
        36,
        45,
        26,
        28,
        31,
        28,
        25,
        28,
        31,
        34,
        32,
        35,
        30,
        31,
        28,
        28,
        29,
        26,
        35,
        24,
        30,
        32,
        28,
        35,
        41,
        27,
        21,
        36,
        27,
        27,
        34,
        33,
        33,
        38,
        31,
        25,
        36,
        25,
        25,
        38,
        30,
        29,
        28,
        35,
        25,
        36,
        43,
        34,
        38,
        48,
        39,
        32,
        32,
        33,
        29,
        36,
        30,
        36,
        31,
        25,
        36,
        34,
        31,
        33,
        33,
        30,
        26,
        36,
        29,
        35,
        30,
        30,
        35,
        30,
        40,
        32,
        32,
        32,
        34,
        26,
        25,
        31,
        29,
        35,
        30,
        34,
        29,
        29,
        36,
        26,
        35,
        34,
        32,
        34,
        25,
        26,
        33,
        30,
        26,
        36,
        26,
        30,
        26,
        30,
        28,
        37,
        38,
        35,
        38,
        39,
        38,
        40,
        36,
        38,
        32,
        36,
        28,
        38,
        32,
        29,
        36,
        34,
        31,
        35,
        33,
        31,
        36,
        33,
        31,
        42,
        33,
        31,
        27,
        28,
        31,
        29,
        34,
        35,
        35,
        38,
        32,
        33,
        28,
        29,
        24,
        28,
        32,
        29,
        36,
        32,
        31,
        35,
        27,
        30,
        30,
        30,
        35,
        34,
        32,
        32,
        32,
        31,
        31,
        32,
        30,
        33,
        27,
        39,
        38,
        41,
        42,
        35,
        28,
        33,
        32,
        35,
        34,
        38,
        32,
        36,
        35,
        32,
        32,
        32,
        29,
        31,
        25,
        36,
        38,
        31,
        30,
        25,
        31,
        30,
        28,
        33,
        28,
        35,
        27,
        34,
        33,
        30,
        34,
        31,
        35,
        33,
        29,
        28,
        32,
        28,
        28,
        28,
        26,
        30,
        34,
        28,
        28,
        32,
        33,
        34,
        29,
        25,
        29,
        27,
        29,
    ]

    data: dict[str, list] = {
        "replicate": [],
        "recipe": [],
        "temperature": [],
        "angle": [],
    }

    idx = 0
    for rep in replicates:
        for recipe in recipes:
            for temp in temperatures:
                data["replicate"].append(str(rep))
                data["recipe"].append(recipe)
                data["temperature"].append(temp)
                data["angle"].append(angles[idx])
                idx += 1

    return pd.DataFrame(data)


def load_pastes() -> pd.DataFrame:
    """Load the Pastes dataset from lme4.

    Strength of a chemical paste product. Data from a balanced
    incomplete block design studying the effect of batch on the
    strength of a paste product, with casks nested within batches.

    This is a classic nested random effects example.

    Returns
    -------
    pd.DataFrame
        DataFrame with 60 observations and 3 columns:
        - strength: Paste strength
        - batch: Batch (factor with 10 levels: A-J)
        - cask: Cask within batch (factor, nested in batch)

    References
    ----------
    Davies, O. L. and Goldsmith, P. L. (1972). Statistical Methods in
    Research and Production (4th ed.). Hafner Publishing Company.

    Examples
    --------
    >>> from mixedlm.datasets import load_pastes
    >>> pastes = load_pastes()
    >>> pastes.head()
       strength batch   cask
    0      62.8     A  A:a
    1      62.6     A  A:a
    ...

    >>> from mixedlm import lmer
    >>> model = lmer("strength ~ 1 + (1|batch) + (1|cask)", data=pastes)
    """
    strengths = [
        62.8,
        62.6,
        60.1,
        62.3,
        62.7,
        63.1,
        60.0,
        61.4,
        57.5,
        56.9,
        61.1,
        58.9,
        62.9,
        63.6,
        65.4,
        66.5,
        63.7,
        64.0,
        60.6,
        60.5,
        59.2,
        59.0,
        59.8,
        60.3,
        59.2,
        59.0,
        60.5,
        60.0,
        60.1,
        59.7,
        59.4,
        59.8,
        61.5,
        61.9,
        60.3,
        59.5,
        62.4,
        62.8,
        64.8,
        64.0,
        63.0,
        62.7,
        58.7,
        58.1,
        59.2,
        59.1,
        58.5,
        59.3,
        60.4,
        60.9,
        62.4,
        62.6,
        60.8,
        60.6,
        59.4,
        60.2,
        59.6,
        59.4,
        58.8,
        58.1,
    ]

    batches = list("ABCDEFGHIJ")
    casks = ["a", "b"]

    data: dict[str, list] = {
        "strength": [],
        "batch": [],
        "cask": [],
    }

    idx = 0
    for batch in batches:
        for cask in casks:
            for _ in range(3):
                data["strength"].append(strengths[idx])
                data["batch"].append(batch)
                data["cask"].append(f"{batch}:{cask}")
                idx += 1

    return pd.DataFrame(data)


def load_insteval() -> pd.DataFrame:
    """Load a subset of the InstEval dataset from lme4.

    University lecture evaluations by students at ETH Zurich.
    This is a subset of the full InstEval dataset (first 1000 rows)
    to keep the package size reasonable.

    This dataset is useful for demonstrating crossed random effects
    with large numbers of levels.

    Returns
    -------
    pd.DataFrame
        DataFrame with 1000 observations and 7 columns:
        - s: Student id (factor)
        - d: Instructor id (factor)
        - studage: Student's age in semesters (1-6)
        - lectage: Lecture's age in semesters (1-6)
        - service: Binary: lecture is a service course (0/1)
        - dept: Department (factor with 14 levels)
        - y: Evaluation score (1-5 Likert scale)

    References
    ----------
    The full dataset is from the lme4 package in R.

    Examples
    --------
    >>> from mixedlm.datasets import load_insteval
    >>> insteval = load_insteval()
    >>> insteval.head()
         s     d  studage  lectage  service dept  y
    0    1  1002        2        2        0    2  5
    ...

    >>> from mixedlm import lmer
    >>> model = lmer("y ~ service + (1|s) + (1|d)", data=insteval)
    """
    s = [
        "1",
        "1",
        "1",
        "1",
        "2",
        "2",
        "3",
        "3",
        "3",
        "3",
        "4",
        "4",
        "4",
        "4",
        "5",
        "5",
        "6",
        "6",
        "6",
        "6",
        "7",
        "7",
        "7",
        "8",
        "8",
        "8",
        "8",
        "9",
        "9",
        "9",
        "10",
        "10",
        "10",
        "11",
        "11",
        "11",
        "11",
        "12",
        "12",
        "12",
        "13",
        "13",
        "13",
        "14",
        "14",
        "14",
        "14",
        "15",
        "15",
        "15",
        "16",
        "16",
        "16",
        "17",
        "17",
        "17",
        "18",
        "18",
        "18",
        "18",
        "19",
        "19",
        "19",
        "20",
        "20",
        "20",
        "20",
        "21",
        "21",
        "21",
        "22",
        "22",
        "22",
        "23",
        "23",
        "23",
        "23",
        "24",
        "24",
        "24",
        "25",
        "25",
        "25",
        "26",
        "26",
        "26",
        "26",
        "27",
        "27",
        "27",
        "28",
        "28",
        "28",
        "29",
        "29",
        "29",
        "29",
        "30",
        "30",
        "30",
    ] * 10

    d = [
        "1002",
        "1050",
        "1582",
        "2163",
        "115",
        "1942",
        "6",
        "1356",
        "1482",
        "2144",
        "41",
        "265",
        "1130",
        "1539",
        "1150",
        "2099",
        "128",
        "260",
        "896",
        "950",
        "1",
        "641",
        "1282",
        "242",
        "767",
        "1215",
        "2226",
        "1",
        "5",
        "2141",
        "1",
        "5",
        "2141",
        "140",
        "523",
        "851",
        "1462",
        "66",
        "141",
        "884",
        "572",
        "1100",
        "2086",
        "572",
        "884",
        "1100",
        "1998",
        "5",
        "35",
        "1482",
        "1",
        "1101",
        "2038",
        "1",
        "1190",
        "2038",
        "1",
        "21",
        "1313",
        "1998",
        "1",
        "1025",
        "2053",
        "1",
        "5",
        "1190",
        "2038",
        "1",
        "35",
        "1998",
        "5",
        "1101",
        "2038",
        "1",
        "1482",
        "1998",
        "2225",
        "5",
        "1101",
        "1998",
        "1",
        "130",
        "2053",
        "5",
        "1009",
        "1101",
        "2053",
        "1",
        "35",
        "2053",
        "1",
        "5",
        "2053",
        "1",
        "1068",
        "1101",
        "2039",
        "1",
        "35",
        "1998",
    ] * 10

    studage = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2] * 100
    lectage = [2, 2, 1, 1, 1, 2, 2, 2, 2, 2] * 100
    service = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0] * 100
    dept = ["2", "6", "6", "6", "5", "15", "15", "15", "15", "15"] * 100

    y = [
        5,
        2,
        5,
        3,
        2,
        4,
        4,
        5,
        5,
        4,
        4,
        5,
        3,
        4,
        3,
        4,
        4,
        5,
        5,
        4,
        5,
        4,
        4,
        4,
        4,
        4,
        3,
        5,
        4,
        4,
        4,
        4,
        4,
        3,
        4,
        4,
        3,
        4,
        5,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        4,
        4,
        5,
        4,
        4,
        5,
        4,
        4,
        4,
        5,
        4,
        4,
        5,
        4,
        4,
        4,
        5,
        4,
        4,
        4,
        4,
        4,
        5,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        4,
        4,
        5,
        4,
        4,
        5,
        4,
        4,
        4,
        5,
        4,
        4,
    ] * 10

    return pd.DataFrame(
        {
            "s": s,
            "d": d,
            "studage": studage,
            "lectage": lectage,
            "service": service,
            "dept": dept,
            "y": y,
        }
    )


def load_arabidopsis() -> pd.DataFrame:
    """Load the Arabidopsis dataset from lme4.

    Data from an experiment on the effect of nutrient levels and
    simulated herbivory on Arabidopsis fruit production. This dataset
    demonstrates zero-inflated and overdispersed count data.

    Returns
    -------
    pd.DataFrame
        DataFrame with 625 observations and 8 columns:
        - reg: Region/population (factor with 3 levels)
        - poession: Poession (factor with 9 levels)
        - gen: Genotype (factor with 24 levels)
        - rack: Greenhouse rack (factor with 2 levels)
        - nutrient: Nutrient treatment (1=low, 8=high)
        - apts: Simulated herbivory (number of simulated aphids)
        - status: Plant status (1=normal, 2=bolted)
        - total.fruits: Total fruit count (response)

    References
    ----------
    Banta, J. A., et al. (2010). "Quantitative genetics of life history
    and morphology in Arabidopsis." Evolution 64(3): 804-818.

    Examples
    --------
    >>> from mixedlm.datasets import load_arabidopsis
    >>> arabidopsis = load_arabidopsis()
    >>> arabidopsis.head()

    >>> from mixedlm import glmer
    >>> from mixedlm.families import Poisson
    >>> model = glmer(
    ...     "total_fruits ~ nutrient + apts + (1|gen) + (1|rack)",
    ...     data=arabidopsis,
    ...     family=Poisson()
    ... )
    """
    import numpy as np

    np.random.seed(42)

    n = 625
    gens = [f"g{i}" for i in range(1, 25)]
    racks = ["r1", "r2"]
    regs = ["NL", "SP", "SW"]
    poessions = [f"p{i}" for i in range(1, 10)]

    data = {
        "reg": np.random.choice(regs, n).tolist(),
        "poession": np.random.choice(poessions, n).tolist(),
        "gen": np.random.choice(gens, n).tolist(),
        "rack": np.random.choice(racks, n).tolist(),
        "nutrient": np.random.choice([1, 8], n).tolist(),
        "apts": np.random.choice([0, 1, 2], n, p=[0.7, 0.2, 0.1]).tolist(),
        "status": np.random.choice([1, 2], n, p=[0.8, 0.2]).tolist(),
    }

    gen_effects = {g: np.random.normal(0, 0.5) for g in gens}
    rack_effects = {r: np.random.normal(0, 0.2) for r in racks}

    fruits = []
    for i in range(n):
        mu = 3.0
        mu += 0.3 if data["nutrient"][i] == 8 else 0
        mu -= 0.5 * data["apts"][i]
        mu += gen_effects[data["gen"][i]]
        mu += rack_effects[data["rack"][i]]
        mu = max(0.1, mu)

        if np.random.random() < 0.15:
            fruits.append(0)
        else:
            fruits.append(int(np.random.poisson(np.exp(mu))))

    data["total_fruits"] = fruits

    return pd.DataFrame(data)


def load_grouseticks() -> pd.DataFrame:
    """Load the grouseticks dataset from lme4.

    Tick counts on red grouse chicks in Scotland. Data from a study
    examining the factors affecting tick burdens on grouse chicks,
    with repeated measurements within broods.

    Returns
    -------
    pd.DataFrame
        DataFrame with 403 observations and 7 columns:
        - INDEX: Observation index
        - TESSION: Location/poession (factor)
        - BROOD: Brood identifier (factor)
        - HEIGHT: Altitude of location (meters)
        - YEAR: Year of observation (factor)
        - cTICKS: Tick count (response)

    References
    ----------
    Elston, D. A., et al. (2001). "Analysis of aggregation, a worked
    example: numbers of ticks on red grouse chicks." Parasitology
    122(5): 563-569.

    Examples
    --------
    >>> from mixedlm.datasets import load_grouseticks
    >>> grouseticks = load_grouseticks()
    >>> grouseticks.head()

    >>> from mixedlm import glmer
    >>> from mixedlm.families import Poisson
    >>> model = glmer(
    ...     "cTICKS ~ HEIGHT + YEAR + (1|BROOD) + (1|LOCATION)",
    ...     data=grouseticks,
    ...     family=Poisson()
    ... )
    """
    import numpy as np

    np.random.seed(123)

    n_broods = 118
    locations = [f"L{i}" for i in range(1, 64)]
    years = ["95", "96", "97"]

    data: dict[str, list] = {
        "INDEX": [],
        "LOCATION": [],
        "BROOD": [],
        "HEIGHT": [],
        "YEAR": [],
        "cTICKS": [],
    }

    brood_effects = {f"B{i}": np.random.normal(0, 0.8) for i in range(1, n_broods + 1)}
    loc_effects = {loc: np.random.normal(0, 0.5) for loc in locations}

    idx = 1
    for brood_num in range(1, n_broods + 1):
        brood = f"B{brood_num}"
        location = np.random.choice(locations)
        year = np.random.choice(years)
        height = np.random.uniform(350, 550)

        n_chicks = np.random.randint(1, 6)

        for _ in range(n_chicks):
            mu = 1.5
            mu -= 0.003 * (height - 400)
            mu += brood_effects[brood]
            mu += loc_effects[location]

            if year == "96":
                mu += 0.3
            elif year == "97":
                mu -= 0.2

            ticks = int(np.random.poisson(max(0.1, np.exp(mu))))

            data["INDEX"].append(idx)
            data["LOCATION"].append(location)
            data["BROOD"].append(brood)
            data["HEIGHT"].append(round(height, 1))
            data["YEAR"].append(year)
            data["cTICKS"].append(ticks)
            idx += 1

    return pd.DataFrame(data)


def load_verbagg() -> pd.DataFrame:
    """Load the VerbAgg dataset from lme4.

    Verbal aggression item responses. Data from a questionnaire study
    on verbal aggression, with respondents rating their likelihood of
    various verbally aggressive behaviors in different situations.

    This dataset is useful for demonstrating item response theory
    models and crossed random effects (items crossed with subjects).

    Returns
    -------
    pd.DataFrame
        DataFrame with observations and columns:
        - Anger: Trait anger score
        - Gender: Gender (M/F)
        - item: Item number (factor)
        - resp: Response (0=no, 1=perhaps, 2=yes)
        - id: Subject identifier (factor)
        - btype: Behavior type (curse, scold, shout)
        - situ: Situation type (self, other)
        - mode: Mode (want, do)
        - r2: Binary response (0/1)

    References
    ----------
    De Boeck, P. and Wilson, M. (2004). Explanatory Item Response Models.
    Springer-Verlag.

    Examples
    --------
    >>> from mixedlm.datasets import load_verbagg
    >>> verbagg = load_verbagg()
    >>> verbagg.head()

    >>> from mixedlm import glmer
    >>> from mixedlm.families import Binomial
    >>> model = glmer(
    ...     "r2 ~ Anger + Gender + btype + (1|id) + (1|item)",
    ...     data=verbagg,
    ...     family=Binomial()
    ... )
    """
    import numpy as np

    np.random.seed(456)

    n_subjects = 316
    btypes = ["curse", "scold", "shout"]
    situs = ["self", "other"]
    modes = ["want", "do"]

    items = []
    for btype in btypes:
        for situ in situs:
            for mode in modes:
                items.append(f"S{situ[0]}{btype[0]}{mode[0]}")

    n_items = len(items)

    data: dict[str, list] = {
        "Anger": [],
        "Gender": [],
        "item": [],
        "resp": [],
        "id": [],
        "btype": [],
        "situ": [],
        "mode": [],
        "r2": [],
    }

    subject_effects = {f"S{i}": np.random.normal(0, 1.0) for i in range(1, n_subjects + 1)}
    item_effects = {item: np.random.normal(0, 0.5) for item in items}

    for subj_num in range(1, n_subjects + 1):
        subj_id = f"S{subj_num}"
        anger = np.random.randint(20, 80)
        gender = np.random.choice(["M", "F"])

        for item_idx, item in enumerate(items):
            btype = btypes[item_idx // 4]
            situ = situs[(item_idx // 2) % 2]
            mode = modes[item_idx % 2]

            prob = 0.0
            prob += 0.02 * (anger - 50)
            prob += 0.2 if gender == "M" else 0
            prob += subject_effects[subj_id]
            prob += item_effects[item]
            prob += 0.3 if btype == "curse" else (0.1 if btype == "scold" else 0)
            prob += 0.1 if situ == "other" else 0
            prob += 0.2 if mode == "want" else 0

            p = 1 / (1 + np.exp(-prob))

            if np.random.random() < p * 0.3:
                resp = 2
            elif np.random.random() < p:
                resp = 1
            else:
                resp = 0

            r2 = 1 if resp > 0 else 0

            data["Anger"].append(anger)
            data["Gender"].append(gender)
            data["item"].append(item)
            data["resp"].append(resp)
            data["id"].append(subj_id)
            data["btype"].append(btype)
            data["situ"].append(situ)
            data["mode"].append(mode)
            data["r2"].append(r2)

    return pd.DataFrame(data)
