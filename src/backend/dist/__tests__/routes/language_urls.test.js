import supertest from "supertest";
import { app } from "../../index.js";
jest.mock("jsonwebtoken", () => ({
    sign: jest.fn(),
    verify: jest.fn().mockReturnValue({
        username: "user_mocked",
    }),
}));
describe("language_urls", () => {
    afterEach(() => {
        jest.clearAllMocks();
    });
    test("should return 400 if no language is passed", async () => {
        const response = await supertest(app)
            .get("/language_urls")
            .set("authorization", "token_mocked");
        expect(response.statusCode).toBe(400);
    });
    test("should return 400 if a wrong language is passed", async () => {
        const response = await supertest(app)
            .get("/language_urls?language=wrong_language")
            .set("authorization", "token_mocked");
        expect(response.statusCode).toBe(400);
    });
    test("should return 200 if the correct language is passed", async () => {
        const response = await supertest(app)
            .get("/language_urls?language=language_mocked")
            .set("authorization", "token_mocked");
        expect(response.statusCode).toBe(200);
    });
});
//# sourceMappingURL=language_urls.test.js.map