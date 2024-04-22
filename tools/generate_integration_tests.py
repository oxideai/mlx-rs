#!/usr/bin/env -S pkgx python@3.10

import typing as t
import mlx.core as mx
import random

emittedTests = {}

random.seed(0)


def new_seed() -> int:
    return random.randint(0, 1000)


def flatten_generator(nested_list):
    for element in nested_list:
        if isinstance(element, list):
            for item in flatten_generator(element):
                yield item
        else:
            yield element


def assert_equal(indent, lhs, rhs, accuracy=None) -> str:
    if accuracy is None:
        return f'{" " * indent}assert_eq!({lhs}, {rhs});\n'
    else:
        return f'{" " * indent}assert_eq!({lhs}, {rhs}, accuracy: {accuracy});\n'


def tuple_to_rust_slice(t) -> str:
    tuple_str = ", ".join([str(i) for i in t])
    return f"&[{tuple_str}]"


def test_name(name) -> str:
    name = name.replace(".", "")
    name = "test_" + name[:1] + name[1:]
    if name in emittedTests:
        count = emittedTests[name]
        count += 1
        emittedTests[name] = count
        name += str(count)
    else:
        emittedTests[name] = 0
    return name


def verify_array(indent, name: str, array: mx.array) -> str:
    result = ""

    shape = array.shape

    if shape == ():
        result += assert_equal(indent, f"{name}.shape().is_empty()", "true")
    else:
        result += assert_equal(indent, f"{name}.shape()", tuple_to_rust_slice(shape))

    dtype = array.dtype
    result += assert_equal(indent, f"{name}.dtype()", "Dtype::" + str(dtype).split(".")[-1].capitalize())

    if dtype == mx.bool_:
        all = mx.all(array).item()
        result += assert_equal(
            indent, f"{name}.all(None, None).item::<bool>()", "true" if all else "false"
        )

        # TODO: Add this check once we support any
        # any = mx.any(array).item()
        # result += assert_equal(
        #     indent, f"{name}.any().item()", "true" if any else "false"
        # )

    else:
        # TODO: update this once we support mean and sum
        # mean = mx.mean(array).item()
        # result += assert_equal(
        #     indent, f"{name}.mean().item(Float.self)", mean, mean * 0.02
        # )
        #
        # sum = mx.sum(array).item()
        # result += assert_equal(
        #     indent, f"{name}.sum().item(Float.self)", sum, sum * 0.02
        # )
        print("Skipping mean and sum tests for now")

    return result


def create_argument(indent, name, value) -> t.Tuple[str, mx.array]:
    if value is None:
        # TODO: Create random array entirely in Rust
        # return f"let {name} = mlx::random::normal([4, 3])", mx.random.normal([4, 3])
        mx_array = mx.random.normal([4, 3])
        flattened_list = list(flatten_generator(mx_array.tolist()))
        return f"let {name} = Array::from_slice(&{flattened_list}, {tuple_to_rust_slice(mx_array.shape)});", mx_array

    if value == "scalar":
        return f"let {name} = MLXRandom.normal()", mx.random.normal()

    if isinstance(value, t.Tuple):
        # TODO: Create uniform array entirely in Rust
        mx_array = mx.random.uniform(0, 1, value)
        flattened_list = list(flatten_generator(mx_array.tolist()))
        return (
            # f"let {name} = mlx::random::uniform(0.0 ..< 1.0, {tuple_to_rust_slice(value)})",
            # mx.random.uniform(0, 1, value),
            f"let {name} = Array::from_slice(&{flattened_list}, {tuple_to_rust_slice(value)});",
            mx_array,
        )

    if isinstance(value, int) or isinstance(value, float):
        return f"let {name} = {value}.into();", value

    if isinstance(value, dict) and "low" in value:
        # TODO: Create uniform array entirely in Rust
        mx_array = mx.random.uniform(value["low"], value["high"], [4, 3])
        flattened_list = list(flatten_generator(mx_array.tolist()))
        return (
            # f"let {name} = mlx::random::uniform(low: {value['low']}, high: {value['high']}, [4, 3])",
            # mx.random.uniform(value["low"], value["high"], [4, 3]),
            f"let {name} = Array::from_slice(&{flattened_list}, &[4, 3]);",
            mx_array,
        )

    if isinstance(value, dict) and "int" in value:
        return (
            f"let {name} = MLXRandom.randInt(low: 0, high: 10, {tuple_to_rust_slice(value['shape'])})",
            mx.random.randint(0, 10, value["shape"]),
        )


def test_operator(
        name: str, op: str, *, rust_name: str = None, lhs=None, rhs=None
) -> str:
    result = ""
    indent = 0
    result += "#[test]\n"
    result += "fn " + test_name(name) + "() {\n"

    seed = new_seed()
    indent += 4
    # TODO: seed the random number generator in Rust
    # result += (" " * indent) + f"MLXRandom.seed({seed})\n"
    mx.random.seed(seed)

    (lhs_decl, lhs) = create_argument(indent, "a", lhs)
    (rhs_decl, rhs) = create_argument(indent, "b", rhs)

    result += (" " * indent) + lhs_decl + "\n"
    if isinstance(lhs, mx.array):
        result += verify_array(indent, "a", lhs)

    result += (" " * indent) + rhs_decl + "\n"
    if isinstance(rhs, mx.array):
        result += verify_array(indent, "b", rhs)

    if rust_name:
        result += (" " * indent) + f"let result = a.{rust_name}(&b);\n"
    else:
        result += (" " * indent) + f"let result = &a {rust_name or op} &b;\n"

    c = eval(f"lhs {op} rhs")

    result += verify_array(indent, "result", c)

    indent -= 4
    result += (" " * indent) + "}\n"
    result += "\n"

    return result


def test_array_function1(
        name: str,
        function_name: str,
        extra="",
        *,
        rust_name: str = None,
        rust_extra="",
        lhs=None,
) -> str:
    # function with 1 array args (self)
    result = ""
    indent = 0
    result += "#[test]\n"
    result += (" " * indent) + "fn " + test_name(name) + "() {\n"

    seed = new_seed()
    indent += 4
    # TODO: seed the random number generator in Rust
    # result += (" " * indent) + f"MLXRandom.seed({seed})\n"
    mx.random.seed(seed)

    (lhs_decl, lhs) = create_argument(indent, "a", lhs)

    result += (" " * indent) + lhs_decl + "\n"
    if isinstance(lhs, mx.array):
        result += verify_array(indent, "a", lhs)

    result += (
                      " " * indent
              ) + f"let result = a.{rust_name or function_name}({rust_extra});\n"

    c = eval(f"lhs.{function_name}({extra})")

    result += verify_array(indent, "result", c)

    indent -= 4
    result += (" " * indent) + "}\n"
    result += "\n"

    return result


def test_free_function1(
        name: str,
        function_name: str,
        extra="",
        *,
        rust_name: str = None,
        rust_extra="",
        lhs=None,
        via_rust_array=False,
) -> str:
    # free function with 1 array arg
    result = ""
    indent = 0
    result += "#[test]\n"
    result += (" " * indent) + "fn " + test_name(name) + "() {\n"

    seed = new_seed()
    indent += 4
    # TODO: seed the random number generator in Rust
    # result += (" " * indent) + f"MLXRandom.seed({seed})\n"
    mx.random.seed(seed)

    (lhs_decl, lhs) = create_argument(indent, "a", lhs)

    result += (" " * indent) + lhs_decl + "\n"
    if isinstance(lhs, mx.array):
        result += verify_array(indent, "a", lhs)

    sep = ", " if len(rust_extra) != 0 else ""
    if via_rust_array:
        result += (
                          " " * indent
                  ) + f"let result = a.{rust_name or function_name}({rust_extra});\n"
    else:
        result += (
                          " " * indent
                  ) + f"let result = {rust_name or function_name}(a{sep}{rust_extra});\n"

    c = eval(f"mx.{function_name}(lhs{sep}{extra})")

    result += verify_array(indent, "result", c)

    indent -= 4
    result += (" " * indent) + "}\n"
    result += "\n"

    return result


def test_array_function2(
        name: str,
        function_name: str,
        extra="",
        *,
        rust_name: str = None,
        rust_extra="",
        lhs=None,
        rhs=None,
) -> str:
    result = ""
    indent = 0
    result += "#[test]\n"
    result += (" " * indent) + "fn " + test_name(name) + "() {\n"

    seed = new_seed()
    indent += 4
    # TODO: seed the random number generator in Rust
    # result += (" " * indent) + f"MLXRandom.seed({seed})\n"
    mx.random.seed(seed)

    (lhs_decl, lhs) = create_argument(indent, "a", lhs)
    (rhs_decl, rhs) = create_argument(indent, "b", rhs)

    result += (" " * indent) + lhs_decl + "\n"
    if isinstance(lhs, mx.array):
        result += verify_array(indent, "a", lhs)

    result += (" " * indent) + rhs_decl + "\n"
    if isinstance(rhs, mx.array):
        result += verify_array(indent, "b", rhs)

    sep = ", " if len(rust_extra) != 0 else ""
    result += (
                      " " * indent
              ) + f"let result = a.{rust_name or function_name}(&b{sep}{rust_extra})\n"

    c = eval(f"lhs.{function_name}(rhs{sep}{extra})")

    result += verify_array(indent, "result", c)

    indent -= 4
    result += (" " * indent) + "}\n"

    return result


def test_free_function2(
        name: str,
        function_name: str,
        extra="",
        *,
        rust_name: str = None,
        rust_extra="",
        lhs=None,
        rhs=None,
        via_rust_array=False,
) -> str:
    # free function with 2 array args
    result = ""
    indent = 0
    result += "#[test]\n"
    result += (" " * indent) + "fn " + test_name(name) + "() {\n"

    seed = new_seed()
    indent += 4
    # TODO: seed the random number generator in Rust
    # result += (" " * indent) + f"MLXRandom.seed({seed})\n"
    mx.random.seed(seed)

    (lhs_decl, lhs) = create_argument(indent, "a", lhs)
    (rhs_decl, rhs) = create_argument(indent, "b", rhs)

    result += (" " * indent) + lhs_decl + "\n"
    if isinstance(lhs, mx.array):
        result += verify_array(indent, "a", lhs)

    result += (" " * indent) + rhs_decl + "\n"
    if isinstance(rhs, mx.array):
        result += verify_array(indent, "b", rhs)

    sep = ", " if len(rust_extra) != 0 else ""
    if via_rust_array:
        result += (
                          " " * indent
                  ) + f"let result = a.{rust_name or function_name}(&b{sep}{rust_extra});\n"
    else:
        result += (
                          " " * indent
                  ) + f"let result = {rust_name or function_name}(a, b{sep}{rust_extra});\n"

    c = eval(f"mx.{function_name}(lhs, rhs{sep}{extra})")

    result += verify_array(indent, "result", c)

    indent -= 4
    result += (" " * indent) + "}\n"

    return result


def generate_integration_tests():
    with open("../tests/integration_test.rs", "w") as f:
        f.write("// Integration tests comparing results vs known results from python\n")
        f.write("// This file is generated by tools/generate_integration_tests.py\n")
        f.write("\n")

        f.write("#![rustfmt::skip]\n")
        f.write("\n")

        f.write("use std::ops::{Add, Div, Mul, Rem, Sub};\n")
        f.write("use num_traits::Pow;\n")
        f.write("use pretty_assertions::assert_eq;\n")
        f.write("use mlx::{Array, Dtype};\n")
        f.write("\n")

        # TODO: test for random seed

        # generate tests for operators

        arithmetic_ops = [
            ("add_op", "+"),
            ("sub_op", "-"),
            ("mul_op", "*"),
            ("div_op", "/"),
            ("mod_op", "%"),
        ]

        for name, op in arithmetic_ops:
            f.write(test_operator(name, op))

        # ** doesn't support a float lhs and requires a particular range
        f.write(
            test_operator(
                "pow_op", "**", rust_name="pow", lhs=dict(low=0.1, high=2.0), rhs=dict(low=0.1, high=2.0)
            )
        )
        f.write(test_operator("pow_op", "**", rust_name="pow", lhs=dict(low=0.1, high=2.0), rhs=1.3))

        logical_ops = [
            ("equal_op", "==", "eq"),
            ("not_equal_op", "!=", "ne"),
            ("less_than_op", "<", "lt"),
            ("less_than_equal_op", "<=", "le"),
            ("greater_than_op", ">", "gt"),
            ("greater_than_equal_op", ">=", "ge"),
        ]

        for name, op, rustop in logical_ops:
            f.write(test_operator(name, op, rust_name=rustop))
            f.write(test_operator(name, op, rust_name=rustop, rhs=1.3))

        # generate tests for single array functions

        array_only_functions = [
            dict(name="abs", array_only=True),
            dict(name="all", array_only=True, axis=True, axes=True, rust_extra="None, None"),
            dict(name="floor", rust_array_only=True),
            dict(name="log", array_only=True, lhs=dict(low=0.1, high=2.0)),
            dict(name="log2", array_only=True, lhs=dict(low=0.1, high=2.0)),
            dict(name="log10", array_only=True, lhs=dict(low=0.1, high=2.0)),
            dict(name="log1p", array_only=True, lhs=dict(low=0.1, high=2.0)),
            dict(name="reciprocal", array_only=True),
            dict(name="round", array_only=True, rust_extra="None"),
            dict(name="sin", array_only=True),
            dict(name="cos", array_only=True),
            dict(name="sqrt", array_only=True, lhs=dict(low=0.1, high=2.0)),
            dict(name="logical_not", rust_array_only=True),
            dict(name="negative", rust_array_only=True, rust_name="neg"),
        ]

        for config in array_only_functions:
            function_name = config["name"]
            rust_name = config.get("rust_name", function_name)
            rust_extra = config.get("rust_extra", "")
            lhs = config.get("lhs", None)

            if "rust_array_only" not in config:
                if "free_only" not in config:
                    f.write(
                        test_array_function1(
                            rust_name, function_name, rust_name=rust_name, rust_extra=rust_extra, lhs=lhs
                        )
                    )
                if "array_only" not in config:
                    f.write(
                        test_free_function1(
                            rust_name, function_name, rust_name=rust_name, rust_extra=rust_extra, lhs=lhs
                        )
                    )
            else:
                f.write(
                    test_free_function1(
                        rust_name, function_name, rust_name=rust_name, rust_extra=rust_extra, lhs=lhs,
                        via_rust_array=True
                    )
                )

            if "axis" in config:
                if "free_only" not in config:
                    f.write(
                        test_array_function1(
                            rust_name,
                            function_name,
                            "axis=-1",
                            rust_extra="&[-1][..], None",
                            lhs=lhs,
                        )
                    )
                if "array_only" not in config:
                    f.write(
                        test_free_function1(
                            rust_name,
                            function_name,
                            "axis=-1",
                            rust_extra="&[-1][..], None",
                            lhs=lhs,
                        )
                    )

            if "axes" in config:
                if "free_only" not in config:
                    f.write(
                        test_array_function1(
                            rust_name,
                            function_name,
                            "axis=[0, -1]",
                            rust_extra="&[0, -1][..], None",
                            lhs=(2, 3, 4, 3),
                        )
                    )
                if "array_only" not in config:
                    f.write(
                        test_free_function1(
                            rust_name,
                            function_name,
                            "axis=[0, -1]",
                            rust_extra="&[0, -1][..], None",
                            lhs=(2, 3, 4, 3),
                        )
                    )

        # generate tests for two array functions

        two_array_functions = [
            dict(name="add", rust_array_only=True),
            dict(name="divide", rust_name="div", rust_array_only=True),
            dict(name="equal", rust_name="eq", rust_array_only=True),
            dict(name="greater", rust_name="gt", rust_array_only=True),
            dict(name="greater_equal", rust_name="ge", rust_array_only=True),
            dict(name="less", rust_name="lt", rust_array_only=True),
            dict(name="less_equal", rust_name="le", rust_array_only=True),
            dict(name="matmul", rust_array_only=True, lhs=(10, 8), rhs=(8, 13)),
            dict(name="multiply", rust_name="mul", rust_array_only=True),
            dict(name="not_equal", rust_name="ne", rust_array_only=True),
            dict(name="remainder", rust_name="rem", rust_array_only=True),
            dict(name="subtract", rust_name="sub", rust_array_only=True),
        ]

        for config in two_array_functions:
            function_name = config["name"]
            rust_name = config.get("rust_name", function_name)
            lhs = config.get("lhs", None)
            rhs = config.get("rhs", None)

            if "rust_array_only" not in config:
                if "free_only" not in config:
                    f.write(
                        test_array_function2(
                            rust_name, function_name, rust_name=rust_name, lhs=lhs, rhs=rhs
                        )
                    )
                f.write(
                    test_free_function2(
                        rust_name, function_name, rust_name=rust_name, lhs=lhs, rhs=rhs
                    )
                )
            else:
                f.write(
                    test_free_function2(
                        rust_name, function_name, rust_name=rust_name, lhs=lhs, rhs=rhs, via_rust_array=True
                    )
                )

        # TODO: Test fft

        # TODO: Test optimizers

        # TODO: Test layers

    print("Integration tests generated successfully")


if __name__ == "__main__":
    generate_integration_tests()
