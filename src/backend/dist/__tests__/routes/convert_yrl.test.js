import supertest from "supertest";
import { app } from "../../index.js";
import axios from "axios";
jest.mock("axios");
const mockedAxios = axios;
jest.mock("jsonwebtoken", () => ({
    sign: jest.fn(),
    verify: jest.fn().mockReturnValue({
        username: "user_mocked",
    }),
}));
describe("convert_yrl", () => {
    afterEach(() => {
        jest.clearAllMocks();
    });
    test("should return 400 when the sentence is not sent", async () => {
        const response = await supertest(app)
            .post("/convert_yrl")
            .set("authorization", "token_mocked");
        expect(response.statusCode).toBe(400);
    });
    test("should return 200 when sending a sentence", async () => {
        mockedAxios.post.mockResolvedValue({
            data: {
                converted_sentence: "converted_sentence_mocked",
            },
        });
        const response = await supertest(app)
            .post("/convert_yrl")
            .send({ sentence: "sentence_mocked" })
            .set("authorization", "token_mocked");
        expect(response.statusCode).toBe(200);
    });
});
//# sourceMappingURL=convert_yrl.test.js.map