type TitleProps = {
  style?: string;
};

export default function Title({ style }: TitleProps) {
  return (
    <h1
      className={`text-lg xl:text-2xl font-bold mb-8 pt-8 ml-4 lg:ml-0 text-center ${style}`}
    >
      Pinimasa yẽgatu rupi
    </h1>
  );
}
