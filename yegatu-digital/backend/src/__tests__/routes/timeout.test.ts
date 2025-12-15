import supertest from "supertest";
import { app } from "../../index.js";
import { data } from "../../shared/data.js";

jest.mock("jsonwebtoken", () => ({
  sign: jest.fn(),
  verify: jest.fn().mockReturnValue({
    username: "user_mocked",
  }),
}));

describe("timeout", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test("should return 200 when calling the route", async () => {
    const response = await supertest(app)
      .get("/timeout")
      .set("authorization", "token_mocked");

    expect(response.statusCode).toBe(200);
  });

  test("should return the data expected", async () => {
    const response = await supertest(app)
      .get("/timeout")
      .set("authorization", "token_mocked");

    expect(response.body).toEqual({
      spell_checker_timeout: data.spellCheckerTimeout,
      translator_timeout: data.translatorTimeout,
    });

    expect(response.statusCode).toBe(200);
  });
});
