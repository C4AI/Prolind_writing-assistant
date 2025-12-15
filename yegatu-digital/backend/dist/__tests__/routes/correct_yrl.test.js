import axios from "axios";
import supertest from "supertest";
import { app } from "../../index.js";
jest.mock("axios");
const mockedAxios = axios;
jest.mock("jsonwebtoken", () => ({
    sign: jest.fn(),
    verify: jest.fn().mockReturnValue({
        username: "user_mocked",
    }),
}));
describe("correct_yrl", () => {
    afterEach(() => {
        jest.clearAllMocks();
    });
    it("should return the sentence corrected", async () => {
        mockedAxios.post.mockResolvedValue({
            data: {
                corrected_words: [
                    "word_mocked",
                    ["word_mocked", "word_mocked", "word_mocked"],
                ],
            },
        });
        const response = await supertest(app)
            .post("/correct_yrl")
            .send({
            sentence_yrl: "sentence_mocked",
            ortography: "orthography_mocked",
            selected_next_words: "word_mocked",
            selected_dict_words: "word_mocked",
            username: "username_mocked",
            token: "token_mocked",
            disable_dic: false,
            disable_next: false,
            disable_word_meaning: false,
        })
            .set("authorization", "token_mocked");
    });
});
//# sourceMappingURL=correct_yrl.test.js.map