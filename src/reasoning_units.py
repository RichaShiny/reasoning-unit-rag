import re


def generate_reasoning_units(question):
    units = [question]
    q = question.lower()

    if "how many people" in q and "arena" in q:
        match = re.search(
            r"the arena where (.+?) played",
            question,
            re.IGNORECASE,
        )

        if match:
            team = match.group(1).strip()
            units.append(f"{team} home arena")
            units.append("arena seating capacity")

    if "where is the company that" in q and "worked for" in q:
        match = re.search(
            r"where is the company that (.+?) worked for",
            question,
            re.IGNORECASE,
        )

        if match:
            person = match.group(1).strip()
            units.append(f"{person} employer")
            units.append("company headquarters")

    if "older" in q or "younger" in q:
        names = re.findall(
            r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)+",
            question,
        )

        for name in names:
            units.append(f"{name} birth date")

    return list(dict.fromkeys(units))


if __name__ == "__main__":
    question = (
        "The arena where the Lewiston Maineiacs played "
        "their home games can seat how many people?"
    )

    print(generate_reasoning_units(question))