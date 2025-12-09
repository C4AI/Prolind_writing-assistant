import supertest from "supertest";
import { app } from "../../index.js";
import axios from "axios";

jest.mock("axios");
const mockedAxios = axios as jest.Mocked<typeof axios>;

jest.mock("jsonwebtoken", () => ({
  sign: jest.fn(),
  verify: jest.fn().mockReturnValue({
    username: "user_mocked",
  }),
}));

describe("translate_en", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test("should return 400 when the orthography is not sent", async () => {
    const response = await supertest(app)
      .post("/translate_en")
      .set("authorization", "token_mocked")
      .send({ sentence: "sentence_mocked" });

    expect(response.statusCode).toBe(400);
  });

  test("should return 400 when the sentence is not sent", async () => {
    const response = await supertest(app)
      .post("/translate_en")
      .set("authorization", "token_mocked")
      .send({ ortography: "orthography_mocked" });

    expect(response.statusCode).toBe(400);
  });

  test("should return the expected response", async () => {
    mockedAxios.post.mockResolvedValue({
      data: {
        translated_sentence: "translated_sentence_mocked",
      },
    });

    const response = await supertest(app)
      .post("/translate_en")
      .set("authorization", "token_mocked")
      .send({
        ortography: "orthography_mocked",
        sentence_en: "sentence_mocked",
      });

    expect(response.body).toStrictEqual({
      sentence_yrl: "translated_sentence_mocked",
    });
  });
});
