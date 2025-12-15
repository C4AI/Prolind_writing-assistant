import supertest from "supertest";
import { app } from "../../index.js"; // Ajustar o caminho se necessário
import axios from "axios";
jest.mock("axios");
const mockedAxios = axios;
jest.mock("jsonwebtoken", () => ({
    sign: jest.fn(),
    verify: jest.fn().mockReturnValue({
        username: "user_mocked",
    }),
}));
describe("translate_yrl_en", () => {
    afterEach(() => {
        jest.clearAllMocks();
    });
    test("should return 400 when the sentence is not sent", async () => {
        mockedAxios.post.mockResolvedValueOnce({
            data: { translated_sentence: "translated_sentence_mocked" },
        });
        const response = await supertest(app)
            .post("/translate_yrl_en")
            .send({
            username: "user_mocked",
            ortography: "orthography_mocked",
            model: "model_mocked",
            disable_dic: false,
            disable_next: false,
            disable_word_meaning: false,
            selected_next_words: "",
            selected_dict_words: "",
        })
            .set("authorization", "token_mocked");
        expect(response.statusCode).toBe(400);
    });
    test("should return 400 when the orthography is not sent", async () => {
        mockedAxios.post.mockResolvedValueOnce({
            data: { translated_sentence: "translated_sentence_mocked" },
        });
        const response = await supertest(app)
            .post("/translate_yrl_en")
            .send({
            username: "user_mocked",
            sentence: "sentence_mocked",
            model: "model_mocked",
            disable_dic: false,
            disable_next: false,
            disable_word_meaning: false,
            selected_next_words: "",
            selected_dict_words: "",
        })
            .set("authorization", "token_mocked");
        expect(response.statusCode).toBe(400);
    });
    test("should return translated sentence if correct parameters are passed", async () => {
        mockedAxios.post.mockResolvedValueOnce({
            data: { translated_sentence: "translated_sentence_mocked" },
        });
        const response = await supertest(app)
            .post("/translate_yrl_en")
            .send({
            username: "user_mocked",
            sentence_yrl: "sentence_mocked",
            ortography: "orthography_mocked",
            model: "model_mocked",
            disable_dic: false,
            disable_next: false,
            disable_word_meaning: false,
            selected_next_words: "",
            selected_dict_words: "",
        })
            .set("authorization", "token_mocked");
        expect(response.statusCode).toBe(200);
        expect(response.body).toEqual({
            sentence_en: "translated_sentence_mocked",
        });
    });
    test("should trim final dot from sentence_yrl", async () => {
        mockedAxios.post.mockResolvedValueOnce({
            data: { translated_sentence: "translated_sentence_mocked" },
        });
        const response = await supertest(app)
            .post("/translate_yrl_en")
            .send({
            username: "user_mocked",
            sentence_yrl: "sentence_mocked.",
            ortography: "orthography_mocked",
            selected_next_words: "",
            selected_dict_words: "",
        })
            .set("authorization", "token_mocked");
        expect(response.statusCode).toBe(200);
        expect(response.body).toEqual({
            sentence_en: "translated_sentence_mocked",
        });
        expect(mockedAxios.post).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({
            sentence: "sentence_mocked",
        }));
    });
});
//# sourceMappingURL=translate_yrl_en.test.js.map