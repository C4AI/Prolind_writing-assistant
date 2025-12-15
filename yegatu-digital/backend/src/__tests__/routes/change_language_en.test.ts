import supertest from "supertest";
import { app } from "../../index.js";

jest.mock("jsonwebtoken", () => ({
  sign: jest.fn(),
  verify: jest.fn().mockReturnValue({
    username: "user_mocked",
  }),
}));

describe("change_language_en", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test("should return 400 if language is not passed", async () => {
    const response = await supertest(app)
      .post("/change_language_en")
      .send({
        username: "user_mocked",
        ortography: "ortography_mocked",
      })
      .set("authorization", "token_mocked");

    expect(response.statusCode).toBe(400);
  });
  test("should return 400 if orthography is not passed", async () => {
    const response = await supertest(app)
      .post("/change_language_en")
      .send({
        username: "user_mocked",
        language: "language_mocked",
      })
      .set("authorization", "token_mocked");

    expect(response.statusCode).toBe(400);
  });
  test("should return 400 if language does not exist", async () => {
    const response = await supertest(app)
      .post("/change_language_en")
      .send({
        username: "user_mocked",
        language: "wrong_language",
        ortography: "orthography_mocked",
      })
      .set("authorization", "token_mocked");

    expect(response.statusCode).toBe(400);
  });
  test("should return 400 if orthography does not exist", async () => {
    const response = await supertest(app)
      .post("/change_language_en")
      .send({
        username: "user_mocked",
        language: "language_mocked",
        ortography: "wrong_orthography",
      })
      .set("authorization", "token_mocked");

    expect(response.statusCode).toBe(400);
  });
  test("should return 200 if all conditions were passed", async () => {
    const response = await supertest(app)
      .post("/change_language_en")
      .send({
        username: "user_mocked",
        language: "language_mocked",
        ortography: "orthography_mocked",
      })
      .set("authorization", "token_mocked");

    expect(response.statusCode).toBe(200);
  });
});
