import supertest from "supertest";
import { app } from "../../index.js";
import { data } from "../../shared/data.js";

jest.mock("jsonwebtoken", () => ({
  sign: jest.fn(),
  verify: jest.fn().mockReturnValue({
    username: "user_mocked",
  }),
}));

describe("next_word_info", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test("should return 200 when calling the route", async () => {
    const response = await supertest(app)
      .get("/next_word_info")
      .set("authorization", "token_mocked");

    expect(response.statusCode).toBe(200);
  });

  test("should return the data expected", async () => {
    const response = await supertest(app)
      .get("/next_word_info")
      .set("authorization", "token_mocked");

    expect(response.body).toEqual({
      next_word_url: data.nextWordUrl,
      next_word_mode: data.nextWordMode,
      next_word_fetch_url: data.fetchUrl,
      next_word_timeout: data.nextWordTimeout,
    });

    expect(response.statusCode).toBe(200);
  });
});
