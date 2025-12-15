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

describe("languages", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test("should return 200 when calling the route", async () => {
    const response = await supertest(app)
      .get("/languages")
      .set("authorization", "token_mocked");

    expect(response.statusCode).toBe(200);
  });

  test("should return the data expected", async () => {
    const response = await supertest(app)
      .get("/languages")
      .set("authorization", "token_mocked");

    expect(response.body).toEqual({
      languages: data.languages,
      language: data.users["user_mocked"].language,
      orthography: data.users["user_mocked"].orthography,
      enable_model_change: data.enableModelChange,
    });

    expect(response.statusCode).toBe(200);
  });
});
