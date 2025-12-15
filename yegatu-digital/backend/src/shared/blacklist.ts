export function blacklist(
  blacklist: string[],
  sentence: string,
  filterType: string
) {
  const words = sentence.split(" ");
  const filteredWords = words.filter((word) => {
    return !blacklist.includes(word);
  });
  if (filterType === "do_nothing") {
    return sentence;
  }
  if (filterType === "remove_word") {
    return filteredWords.join(" ");
  }
  if (filterType === "remove_sentence") {
    return "";
  }
  if (filterType === "redact_word") {
    return words
      .map((word) => {
        if (blacklist.includes(word)) {
          return "*******";
        }
        return word;
      })
      .join(" ");
  } else {
    throw new Error("Invalid filter type");
  }
}
