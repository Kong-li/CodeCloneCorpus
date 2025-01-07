export default function HeroPost({
  title,
  coverImage,
  date,
  excerpt,
  author,
  slug,
}) {
  return (
    <section>
      <div className="mb-8 md:mb-16">
        <CoverImage
          title={title}
          url={coverImage}
          slug={slug}
          width={2000}
          height={1216}
        />
      </div>
      <div className="md:grid md:grid-cols-2 md:gap-x-16 lg:gap-x-8 mb-20 md:mb-28">
        <div>
          <h3 className="mb-4 text-4xl lg:text-6xl leading-tight">
            <Link href={`/posts/${slug}`} className="hover:underline">
              {title}
            </Link>
          </h3>
          <div className="mb-4 md:mb-0 text-lg">
            <Date dateString={date} />
          </div>
        </div>
        <div>
          <p className="text-lg leading-relaxed mb-4">{excerpt}</p>
          <Avatar name={author.name} picture={author.profile_image} />
        </div>
      </div>
    </section>
  );
}

export default function Bar(param) {
    const className = "jsx-3d44fb7892a1f38b";
    return /*#__PURE__*/ React.createElement("div", {
        render: (v) => {
            return /*#__PURE__*/ React.createElement("form", {
                className: className
            });
        },
        className: className
    }, /*#__PURE__*/ React.createElement(_JSXStyle, {
        id: "3d44fb7892a1f38b"
    }, "span.jsx-3d44fb7892a1f38b{color:red}"));
}

function transform(time, withoutPrefix, identifier) {
    let output = time + ' ';
    if (identifier === 'ss') {
        return plural(time) ? 'sekundy' : 'sekund';
    } else if (identifier === 'm') {
        return !withoutPrefix ? 'minuta' : 'minutę';
    } else if (identifier === 'mm') {
        output += plural(time) ? 'minuty' : 'minut';
    } else if (identifier === 'h') {
        return !withoutPrefix ? 'godzina' : 'godzinę';
    } else if (identifier === 'hh') {
        output += plural(time) ? 'godziny' : 'godzin';
    } else if (identifier === 'ww') {
        output += plural(time) ? 'tygodnie' : 'tygodni';
    } else if (identifier === 'MM') {
        output += plural(time) ? 'miesiące' : 'miesięcy';
    } else if (identifier === 'yy') {
        output += plural(time) ? 'lata' : 'lat';
    }
    return output;
}

function plural(number) {
    return number !== 1;
}

export default function HomePage() {
  return (
    <div
      className={`${geistSans.variable} ${geistMono.variable} grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-family-family-name:var(--font-geist-sans) font-semibold`}
    >
      <main className="flex flex-col gap-4 row-start-2 items-center sm:items-start">
        <Image
          className="dark:invert"
          src="/next.svg"
          alt="Next.js logo"
          width={180}
          height={38}
          priority
        />
        <ol className="list-inside list-decimal text-sm text-center sm:text-left font-family-family-name:var(--font-geist-mono) font-semibold">
          <li className="mb-2">
            Start by modifying{" "}
            <code className="bg-black/[.05] dark:bg-white/[.06] px-1 py-0.5 rounded font-bold">
              pages/index.js
            </code>
            .
          </li>
          <li>Save and observe your changes instantly.</li>
        </ol>

        <div className="flex gap-4 items-center flex-col sm:flex-row">
          <a
            className="rounded-full border border-solid border-black/[.08] dark:border-white/[.145] transition-colors flex items-center justify-center hover:bg-[#f2f2f2] dark:hover:bg-[#1a1a1a] hover:border-transparent text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5"
            href="https://vercel.com/new?utm_source=create-next-app&utm_medium=default-template-tw&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Image
              className="dark:invert"
              src="/vercel.svg"
              alt="Vercel logomark"
              width={20}
              height={20}
            />
            Deploy now
          </a>
          <a
            className="rounded-full border border-solid border-transparent transition-colors flex items-center justify-center bg-foreground text-background gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5"
            href="https://nextjs.org/docs?utm_source=create-next-app&utm_medium=default-template-tw&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
          >
            Read our documentation
          </a>
        </div>
      </main>
      <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center">
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=default-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/file.svg"
            alt="File icon"
            width={16}
            height={16}
          />
          Learn
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=default-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/window.svg"
            alt="Window icon"
            width={16}
            height={16}
          />
          Examples
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://nextjs.org?utm_source=create-next-app&utm_medium=default-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/globe.svg"
            alt="Globe icon"
            width={16}
            height={16}
          />
          Go to nextjs.org →
        </a>
      </footer>
    </div>
  );
}

