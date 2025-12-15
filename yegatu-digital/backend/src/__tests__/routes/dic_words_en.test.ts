import supertest from "supertest";
import { app } from "../../index.js";
import { data } from "../../shared/data.js";
import { dataMocked, dataMockedResponse } from "../../setup_tests.js";

jest.mock("jsonwebtoken", () => ({
  sign: jest.fn(),
  verify: jest.fn().mockReturnValue({
    username: "user_mocked",
  }),
}));

describe("dic_words_en", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test("should return 200 when calling the route", async () => {
    const response = await supertest(app)
      .post("/dic_words_en")
      .set("authorization", "token_mocked").send({
        language: "language_mocked",
        ortography: "orthography_mocked",
      });

    expect(response.statusCode).toBe(200);
  });

  test("should return the data expected", async () => {
    const response = await supertest(app)
      .post("/dic_words_en")
      .set("authorization", "token_mocked").send({
        language: "language_mocked",
        ortography: "orthography_mocked",
      });

    expect(response.body).toEqual({
      dic_words: dataMockedResponse.dicListObjEn,
      char_conv_map: dataMockedResponse.charConvMap,
    });

    expect(response.statusCode).toBe(200);
  });
});
