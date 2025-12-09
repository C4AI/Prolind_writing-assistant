import { forwardRef, LegacyRef } from "react";

type InputProps = {
    type: string;
    placeholder: string;
};

const Input = forwardRef(
    ({ type, placeholder }: InputProps, ref: LegacyRef<HTMLInputElement>) => {
        return (
            <input
                type={type}
                required
                ref={ref}
                placeholder={placeholder}
                className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
        );
    }
);

export default Input;
