type ErrorMessageProps = {
    message: string;
};

export default function ErrorMessage({ message }: ErrorMessageProps) {
    return (
        <>
            {message && (
                <h1 className="text-red-500 font-bold text-xl mt-10">
                    {message}
                </h1>
            )}
        </>
    );
}
