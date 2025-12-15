import supertest from "supertest";
import { app } from "../../index.js";
import { data } from "../../shared/data.js";
import { dataMocked } from "../../setup_tests.js";

jest.mock("jsonwebtoken", () => ({
  sign: jest.fn(),
  verify: jest.fn().mockReturnValue({
    username: "user_mocked",
  }),
}));

describe("config", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test("should return 200 when calling the route", async () => {
    const response = await supertest(app)
      .post("/config")
      .set("authorization", "token_mocked");

    expect(response.statusCode).toBe(200);
  });

  test("should update all data properties", async () => {
    await supertest(app).post("/config").set("authorization", "token_mocked");

    var charConvMap: [string, string][] = [];
    for (let dest_char in dataMocked.char_map) {
      const sourceCharList: string[] = (
        dataMocked.char_map as { [key: string]: string[] }
      )[dest_char];
      for (var i = 0; i < sourceCharList.length; i++) {
        let map: [string, string] = [sourceCharList[i], dest_char];
        charConvMap.push(map);
        map = [sourceCharList[i].toUpperCase(), dest_char.toUpperCase()];
        charConvMap.push(map);
      }
    }

    expect(data.charConvMap).toStrictEqual(charConvMap);
    expect(data.nextWordMode).toStrictEqual(dataMocked.next_word_mode);
    expect(data.nextWordTimeout).toStrictEqual(dataMocked.next_word_timeout);
    expect(data.userAuthTimeout).toStrictEqual(dataMocked.user_auth_timeout);
    expect(data.tokenKey).toStrictEqual(dataMocked.auth_token_key);
    expect(data.users).toStrictEqual(dataMocked.users);
    expect(data.languages).toStrictEqual(dataMocked.languages);
    expect(data.spellCheckerTimeout).toStrictEqual(
      dataMocked.spell_checker_timeout
    );
    expect(data.translatorTimeout).toStrictEqual(dataMocked.translator_timeout);
    expect(data.enableModelChange).toStrictEqual(
      dataMocked.enable_model_change
    );
    expect(data.fetchUrl).toStrictEqual(dataMocked.next_word_fetch_url);
  });
});
