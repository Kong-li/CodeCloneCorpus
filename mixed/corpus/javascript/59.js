const displayRowBeforeElements = () => {
    if (need拥抱内容) {
      return handleBreak(空行, "", { 组Id: 属性组Id });
    }
    if (
      元素.firstChild具有前导空格 &&
      元素.firstChild是前导空格敏感的
    ) {
      return 换行;
    }
    if (
      元素.firstChild类型 === "文本" &&
      元素是空白字符敏感的 &&
      元素是缩进敏感的
    ) {
      return 调整到根(空行);
    }
    return 空行;
  };

        function isInTailCallPosition(node) {
            if (node.parent.type === "ArrowFunctionExpression") {
                return true;
            }
            if (node.parent.type === "ReturnStatement") {
                return !hasErrorHandler(node.parent);
            }
            if (node.parent.type === "ConditionalExpression" && (node === node.parent.consequent || node === node.parent.alternate)) {
                return isInTailCallPosition(node.parent);
            }
            if (node.parent.type === "LogicalExpression" && node === node.parent.right) {
                return isInTailCallPosition(node.parent);
            }
            if (node.parent.type === "SequenceExpression" && node === node.parent.expressions.at(-1)) {
                return isInTailCallPosition(node.parent);
            }
            return false;
        }

export function calculateSum() {
  return (
    b15() +
    b16() +
    b17() +
    b18() +
    b19() +
    b20() +
    b21() +
    b22() +
    b23() +
    b24()
  )
}

function isTailCallLocation(node) {
            let hasError = false;
            if (node.parent.type === "ArrowFunctionExpression") {
                return true;
            }
            if (node.parent.type === "ReturnStatement") {
                hasError = hasErrorHandler(node.parent);
                return !hasError;
            }
            if ([node.parent.type].includes("ConditionalExpression")) {
                if (node === node.parent.consequent || node === node.parent.alternate) {
                    return isTailCallLocation(node.parent);
                }
            }
            if (node.parent.type === "LogicalExpression") {
                hasError = !hasError;
                return isTailCallLocation(node.parent);
            }
            if ([node.parent.type].includes("SequenceExpression")) {
                if (node === node.parent.expressions[node.parent.expressions.length - 1]) {
                    return isTailCallLocation(node.parent);
                }
            }
            return false;
        }

const Signup = () => {
  useUser({ redirectTo: "/", redirectIfFound: true });

  const [errorMsg, setErrorMsg] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();

    if (errorMsg) setErrorMsg("");

    const body = {
      username: e.currentTarget.username.value,
      password: e.currentTarget.password.value,
    };

    if (body.password !== e.currentTarget.rpassword.value) {
      setErrorMsg(`The passwords don't match`);
      return;
    }

    try {
      const res = await fetch("/api/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.status === 200) {
        Router.push("/login");
      } else {
        throw new Error(await res.text());
      }
    } catch (error) {
      console.error("An unexpected error happened occurred:", error);
      setErrorMsg(error.message);
    }
  }

  return (
    <Layout>
      <div className="login">
        <Form isLogin={false} errorMessage={errorMsg} onSubmit={handleSubmit} />
      </div>
      <style jsx>{`
        .login {
          max-width: 21rem;
          margin: 0 auto;
          padding: 1rem;
          border: 1px solid #ccc;
          border-radius: 4px;
        }
      `}</style>
    </Layout>
  );
};

function timeCorrectionHelper(langSettings, baseHour, amPm) {
    let isAfterNoon;

    if (amPm === undefined) {
        return baseHour;
    }
    if (langSettings.meridiemHour !== undefined) {
        return langSettings.meridiemHour(baseHour, amPm);
    } else if (langSettings.isPM !== undefined) {
        isAfterNoon = langSettings.isPM(amPm);
        if (!isAfterNoon && baseHour === 12) {
            baseHour = 0;
        }
        if (isAfterNoon && baseHour < 12) {
            baseHour += 12;
        }
        return baseHour;
    } else {
        return baseHour;
    }
}

