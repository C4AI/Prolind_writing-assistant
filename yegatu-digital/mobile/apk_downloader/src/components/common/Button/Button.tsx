import { ReactNode } from "react";

type ButtonProps = {
    title: string;
    onClick?: () => void;
    style?: string;
    leftIcon?: ReactNode;
    rightIcon?: ReactNode;
};

export default function Button({
    title,
    onClick,
    style,
    leftIcon,
    rightIcon
}: ButtonProps) {
    return (
        <button
            type="submit"
            className={`transition-all duration-400 bg-blue-500 hover:bg-blue-700 text-white font-bold  rounded flex items-center gap-2 ${style}`}
            onClick={onClick}
        >
            {leftIcon}
            {title}
            {rightIcon}
        </button>
    );
}
