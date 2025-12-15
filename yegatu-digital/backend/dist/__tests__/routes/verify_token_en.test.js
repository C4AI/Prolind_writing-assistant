import supertest from "supertest";
import { app } from "../../index.js";
import jwt from "jsonwebtoken";
jest.mock("jsonwebtoken");
const mockedJwt = jwt;
jest.mock("jsonwebtoken", () => ({
    sign: jest.fn(),
    verify: jest.fn().mockReturnValue({
        username: "user_mocked",
    }),
}));
describe("verify_token_en", () => {
    afterEach(() => {
        jest.clearAllMocks();
    });
    test("should return 401 when the token is not sent", async () => {
        const response = await supertest(app).get("/verify_token_en");
        expect(response.statusCode).toBe(401);
    });
    test("should return 401 when the token is invalid", async () => {
        mockedJwt.verify.mockImplementation((token, key, callback) => {
            callback(new Error("Invalid token"));
        });
        const response = await supertest(app)
            .get("/verify_token_en")
            .set("authorization", "invalid_token");
        expect(response.statusCode).toBe(401);
    });
    test("should accept if token is valid", async () => {
        mockedJwt.verify.mockImplementation((token, key, callback) => {
            callback(null, { username: "user_mocked" });
        });
        const response = await supertest(app)
            .get("/verify_token_en")
            .set("authorization", "valid_token");
        expect(response.statusCode).toBe(200);
    });
});
//# sourceMappingURL=verify_token_en.test.js.map