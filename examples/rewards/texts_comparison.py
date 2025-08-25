import math
import re

# fmt: off
# Stopwords set for vocabulary analysis
STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "if", "because", "as", "what", "which",
    "this", "that", "these", "those", "then", "just", "so", "than", "such", "when",
    "who", "how", "where", "why", "is", "am", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing", "to",
    "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once", "here", "there",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "should", "now", "of"
}
# fmt: on


def proper_length_ratio_reward(
    completions1, completions2, target_min=2.4, target_max=2.8
):
    """Reward function that gives high reward when the second completion is 2.4-2.8 times longer than the first.

    The maximum reward is given when the ratio is exactly in the target range (2.4-2.8x),
    and gradually decreases as the ratio moves further from this range.

    Args:
        completions1: List of text completions from agent 1
        completions2: List of text completions from agent 2

    Returns:
        List of rewards between 0.0 and 1.0:
        - 1.0: Perfect ratio (length2/length1 between 2.4-2.8)
        - >0.0 to <1.0: Partial reward that decreases exponentially as the ratio deviates from target range
        - 0.0: Empty first completion or extremely poor ratio
    """
    rewards = []
    for c1, c2 in zip(completions1, completions2):
        len1, len2 = len(c1), len(c2)

        if len1 == 0:
            rewards.append(0.0)
            continue

        ratio = len2 / len1

        if target_min <= ratio <= target_max:
            reward = 1.0
        else:
            if ratio < target_min:
                distance = target_min - ratio
            else:
                distance = ratio - target_max

            reward = math.exp(-distance)

        rewards.append(float(reward))

    return rewards


def vocabulary_richness_reward(completions1, completions2):
    """Reward function that gives high reward when the second completion has higher
    vocabulary richness (Type-Token Ratio without stopwords) than the first.

    The reward is based on the improvement in TTR from the first to the second completion.
    Maximum reward is given when the second completion's TTR is substantially higher,
    and gradually decreases as the improvement diminishes.

    Args:
        completions1: List of text completions from agent 1
        completions2: List of text completions from agent 2

    Returns:
        List of rewards between 0.0 and 1.0:
        - 1.0: Perfect improvement (TTR2/TTR1 >= 2.0 or TTR2 > 0 when TTR1 = 0)
        - >0.0 to <1.0: Partial reward based on the improvement ratio
        - 0.0: No improvement or both completions have zero TTR
    """

    def calculate_ttr(text, stopwords):
        """Calculate Type-Token Ratio (TTR) excluding stopwords.

        Args:
            text: String text to analyze
            stopwords: Set of stopwords to exclude

        Returns:
            Float value representing TTR (unique content words / total content words)
        """
        words = re.findall(r"\b\w+\b", text.lower())

        if stopwords:
            content_words = [word for word in words if word not in stopwords]
        else:
            content_words = words

        if not content_words:
            return 0.0

        types = len(set(content_words))
        tokens = len(content_words)

        return types / tokens if tokens > 0 else 0.0

    vocabulary_richness_reward.calculate_ttr = calculate_ttr
    rewards = []
    for c1, c2 in zip(completions1, completions2):
        ttr1 = calculate_ttr(c1, STOPWORDS)
        ttr2 = calculate_ttr(c2, STOPWORDS)

        if ttr1 == 0:
            if ttr2 > 0:
                reward = 1.0
            else:
                reward = 0.0
        else:
            improvement = ttr2 / ttr1

            target_min = 1.2
            target_max = 2.0

            if improvement >= target_max:
                reward = 1.0
            elif improvement >= target_min:
                reward = (improvement - target_min) / (target_max - target_min)
            else:
                distance = target_min - improvement
                reward = math.exp(-2 * distance)

        rewards.append(float(reward))

    return rewards
