import supertest from "supertest";
import { app } from "../../index.js";

jest.mock("jsonwebtoken", () => ({
  sign: jest.fn(),
  verify: jest.fn().mockReturnValue({
    username: "user_mocked",
  }),
}));

describe("auth_en", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test("should reject if app_id is not in user data", async () => {
    const response = await supertest(app).post("/auth_en").send({
      username: "user_mocked",
      password: "password_mocked",
      app_id: 100,
    });

    expect(response.statusCode).toBe(401);
  });
  test("should reject if the password is wrong", async () => {
    const response = await supertest(app).post("/auth_en").send({
      username: "user_mocked",
      password: "wrong_password",
      app_id: 1,
    });

    expect(response.statusCode).toBe(401);
  });
  test("should reject if the user is not in the database", async () => {
    const response = await supertest(app).post("/auth_en").send({
      username: "wrong_user",
      password: "password_mocked",
      app_id: 1,
    });

    expect(response.statusCode).toBe(401);
  });

  test("should accept if the user is correct", async () => {
    const response = await supertest(app).post("/auth_en").send({
      username: "user_mocked",
      password: "password_mocked",
      app_id: 1,
    });

    expect(response.statusCode).toBe(200);
  });
});
