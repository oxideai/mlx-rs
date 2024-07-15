#!/usr/bin/env -S pkgx python@3.11

import typing as t
import mlx.core as mx
import random
import subprocess

emittedTests = {}

random.seed(0)


def new_seed() -> int:
    return random.randint(0, 1000)


def assert_equal(indent, lhs, rhs, accuracy=None) -> str:
    if accuracy is None:
        return f'{" " * indent}assert_eq!({lhs}, {rhs});\n'
    else:
        return f'{" " * indent}float_eq!({lhs}, {rhs}, abs <= {accuracy});\n'


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
            indent, f"{name}.all(None, None).unwrap().item::<bool>()", "true" if all else "false"
        )

        any = mx.any(array).item()
        result += assert_equal(
            indent, f"{name}.any(None, None).unwrap().item::<bool>()", "true" if any else "false"
        )

    else:
        mean = mx.mean(array).item()
        result += assert_equal(
            indent, f"{name}.mean(None, None).unwrap().item::<f32>()", mean, mean * 0.02
        )

        sum = mx.sum(array).item()
        result += assert_equal(
            indent, f"{name}.sum(None, None).unwrap().item::<f32>()", sum, sum * 0.02
        )

    return result


def create_argument(indent, name, value) -> t.Tuple[str, mx.array]:
    if value is None:
        return f"let {name} = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();", mx.random.normal([4, 3])

    if value == "scalar":
        return f"let {name} = mlx_rs::random::normal::<f32>(None, None, None, None).unwrap();", mx.random.normal()

    if isinstance(value, t.Tuple):
        return (
            f"let {name} = mlx_rs::random::uniform::<_, f32>(0.0f32, 1.0f32, {tuple_to_rust_slice(value)}, None).unwrap();",
            mx.random.uniform(0, 1, value),
        )

    if isinstance(value, int) or isinstance(value, float):
        return f"let {name}: Array = {value}.into();", value

    if isinstance(value, dict) and "low" in value:
        return (
            f"let {name} = mlx_rs::random::uniform::<_, f32>({value['low']}f32, {value['high']}f32, &[4, 3], None).unwrap();",
            mx.random.uniform(value["low"], value["high"], [4, 3]),
        )

    if isinstance(value, dict) and "int" in value:
        return (
            f"let {name} = mlx_rs::random::randint::<i32>(0, 10, {tuple_to_rust_slice(value['shape'])}).unwrap();",
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
    result += (" " * indent) + f"mlx_rs::random::seed({seed});\n"
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
        # if its pow, we don't unwrap the result
        if rust_name == "pow":
            result += (" " * indent) + f"let result = a.{rust_name}(&b);\n"
        else:
            result += (" " * indent) + f"let result = a.{rust_name}(&b).unwrap();\n"
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
    result += (" " * indent) + f"mlx_rs::random::seed({seed});\n"
    mx.random.seed(seed)

    (lhs_decl, lhs) = create_argument(indent, "a", lhs)

    result += (" " * indent) + lhs_decl + "\n"
    if isinstance(lhs, mx.array):
        result += verify_array(indent, "a", lhs)

    nounwrap = ["abs", "sqrt", "cos", "sin", "round", "reciprocal", "log1p", "log10", "log2", "log"]
    if rust_name in nounwrap:
        result += (
                      " " * indent
              ) + f"let result = a.{rust_name or function_name}({rust_extra});\n"
    else:
        result += (
                      " " * indent
              ) + f"let result = a.{rust_name or function_name}({rust_extra}).unwrap();\n"

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
    result += (" " * indent) + f"mlx_rs::random::seed({seed});\n"
    mx.random.seed(seed)

    (lhs_decl, lhs) = create_argument(indent, "a", lhs)

    result += (" " * indent) + lhs_decl + "\n"
    if isinstance(lhs, mx.array):
        result += verify_array(indent, "a", lhs)

    sep = ", " if len(rust_extra) != 0 else ""
    if via_rust_array:
        nounwrap = ["logical_not"]
        if rust_name in nounwrap:
            result += (
                            " " * indent
                    ) + f"let result = a.{rust_name or function_name}({rust_extra});\n"
        else:
            result += (
                          " " * indent
                  ) + f"let result = a.{rust_name or function_name}({rust_extra}).unwrap();\n"
    else:
        result += (
                          " " * indent
                  ) + f"let result = {rust_name or function_name}(&a{sep}{rust_extra});\n"

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
    result += (" " * indent) + f"mlx_rs::random::seed({seed});\n"
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
    result += "\n"

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
    result += (" " * indent) + f"mlx_rs::random::seed({seed});\n"
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
        nounwrap = ["mul", "rem", "sub", "div", "add"]
        if rust_name in nounwrap:
            result += (
                            " " * indent
                    ) + f"let result = a.{rust_name or function_name}(&b{sep}{rust_extra});\n"
        else:
            result += (
                          " " * indent
                  ) + f"let result = a.{rust_name or function_name}(&b{sep}{rust_extra}).unwrap();\n"
    else:
        result += (
                          " " * indent
                  ) + f"let result = {rust_name or function_name}(a, b{sep}{rust_extra});\n"

    c = eval(f"mx.{function_name}(lhs, rhs{sep}{extra})")

    result += verify_array(indent, "result", c)

    indent -= 4
    result += (" " * indent) + "}\n"
    result += "\n"

    return result


def test_fft(
        function_name: str, n=None, s=None, axis=None, axes=None, *, value=None
) -> str:
    result = ""
    indent = 0
    result += "#[test]\n"
    result += (" " * indent) + "fn " + test_name(function_name + "_") + "() {\n"

    seed = new_seed()
    indent += 4
    result += (" " * indent) + f"mlx_rs::random::seed({seed});\n"
    mx.random.seed(seed)

    (r_decl, r) = create_argument(indent, "r", value)
    (i_decl, i) = create_argument(indent, "i", value)

    result += (" " * indent) + r_decl + "\n"
    if isinstance(r, mx.array):
        result += verify_array(indent, "r", r)

    result += (" " * indent) + i_decl + "\n"
    if isinstance(i, mx.array):
        result += verify_array(indent, "i", i)

    # combine into a complex array
    result += (" " * indent) + f"let c: Array = &(&r + &i) * &Array::from_complex(Complex32::new(0., 1.));\n"
    result += (" " * indent) + f"assert_eq!(c.dtype(), Dtype::Complex64);\n"
    c = r + 1j * i

    e = f"mx.fft.{function_name}(c"
    result += (" " * indent) + f"let result = {function_name}_device(&c"
    if n is not None:
        result += f", {n}"
        e += f", n=n"
    if s is not None:
        result += f", &{s}[..]"
        e += f", s=s"
    if n is None and s is None:
        result += f", None"
    if axis is not None:
        result += f", {axis}"
        e += f", axis=axis"
    if axes is not None:
        result += f", {tuple_to_rust_slice(axes)}[..]"
        e += f", axes=axes"
    if axis is None and axes is None:
        result += f", None"

    result += ", StreamOrDevice::cpu()).unwrap();\n"
    e += ", stream=mx.cpu)"

    c = eval(e)

    if c.dtype == mx.complex64:
        # split back out real and imaginary
        result += (" " * indent) + f"let result_real = result.as_type::<f32>();\n"
        result += (" " * indent) + f"let result_imaginary = (&result / &Array::from_complex(Complex32::new(0., 1.))).as_type::<f32>();\n"

        r = c.astype(mx.float32)
        i = (c / 1j).astype(mx.float32)

        result += verify_array(indent, "result_real", r)
        result += verify_array(indent, "result_imaginary", i)
    else:
        result += verify_array(indent, "result", c)

    indent -= 4
    result += (" " * indent) + "}\n"
    result += "\n"

    return result


def generate_integration_tests():
    with open("../tests/integration_test.rs", "w") as f:
        f.write("// Integration tests comparing results vs known results from python\n")
        f.write("// This file is @generated by tools/generate_integration_tests.py\n")
        f.write("\n")

        f.write("use std::ops::{Add, Div, Mul, Rem, Sub};\n")
        f.write("use num_traits::Pow;\n")
        f.write("use pretty_assertions::assert_eq;\n")
        f.write("use num_complex::Complex32;\n")
        f.write("use mlx_rs::{Array, Dtype, StreamOrDevice, ops::{indexing::{argmax}, acos}, fft::{fft_device, ifft_device, rfft_device, irfft_device, fft2_device, ifft2_device, fftn_device, ifftn_device, rfft2_device, irfft2_device, rfftn_device, irfftn_device}};\n")
        f.write("use float_eq::float_eq;\n")
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
            dict(name="all", array_only=True, axis=dict(rust_extra="&[-1][..], None"), axes=dict(rust_extra="&[0, -1][..], None"), rust_extra="None, None"),
            dict(name="any", array_only=True, axis=dict(rust_extra="&[-1][..], None"), axes=dict(rust_extra="&[0, -1][..], None"), rust_extra="None, None"),
            # dict(name="argmax", non_axis_name="argmax_all", free_only=True, axis=True, rust_extra="None"),
            # dict(name="argmin", free_only=True, axis=True, rust_extra="None"),
            dict(name="cummax", array_only=True, axis=dict(rust_extra="-1, None, None"), rust_extra="None, None, None"),
            dict(name="cummin", array_only=True, axis=dict(rust_extra="-1, None, None"), rust_extra="None, None, None"),
            dict(name="cumprod", array_only=True, axis=dict(rust_extra="-1, None, None"), rust_extra="None, None, None"),
            dict(name="cumsum", array_only=True, axis=dict(rust_extra="-1, None, None"), rust_extra="None, None, None"),
            dict(name="expand_dims", no_bare=True, rust_array_only=True, axis=dict(rust_extra="&[-1][..]"), axes=dict(rust_extra="&[0, -1][..]")),
            dict(name="floor", rust_array_only=True),
            dict(name="log", array_only=True, lhs=dict(low=0.1, high=2.0)),
            dict(name="log2", array_only=True, lhs=dict(low=0.1, high=2.0)),
            dict(name="log10", array_only=True, lhs=dict(low=0.1, high=2.0)),
            dict(name="log1p", array_only=True, lhs=dict(low=0.1, high=2.0)),
            dict(name="logsumexp", rust_name="log_sum_exp", array_only=True, axis=dict(rust_extra="&[-1][..], None"), axes=dict(rust_extra="&[0, -1][..], None"), rust_extra="None, None"),
            dict(name="max", array_only=True, axis=dict(rust_extra="&[-1][..], None"), axes=dict(rust_extra="&[0, -1][..], None"), rust_extra="None, None"),
            dict(name="mean", array_only=True, axis=dict(rust_extra="&[-1][..], None"), axes=dict(rust_extra="&[0, -1][..], None"), rust_extra="None, None"),
            dict(name="min", array_only=True, axis=dict(rust_extra="&[-1][..], None"), axes=dict(rust_extra="&[0, -1][..], None"), rust_extra="None, None"),
            dict(name="prod", array_only=True, axis=dict(rust_extra="&[-1][..], None"), axes=dict(rust_extra="&[0, -1][..], None"), rust_extra="None, None"),
            dict(name="reciprocal", array_only=True),
            dict(name="round", array_only=True, rust_extra="None"),
            dict(name="sin", array_only=True),
            dict(name="cos", array_only=True),
            dict(name="sqrt", array_only=True, lhs=dict(low=0.1, high=2.0)),
            dict(name="sum", array_only=True, axis=dict(rust_extra="&[-1][..], None"), axes=dict(rust_extra="&[0, -1][..], None"), rust_extra="None, None"),
            dict(name="var", rust_name="variance", array_only=True, axis=dict(rust_extra="&[-1][..], None, None"), axes=dict(rust_extra="&[0, -1][..], None, None"), rust_extra="None, None, None"),
            # free functions only
            # dict(name="arccos", rust_name="acos", free_only=True, lhs=dict(low=0.1, high=2.0)),
            dict(name="logical_not", rust_array_only=True),
            dict(name="negative", rust_array_only=True),
        ]

        for config in array_only_functions:
            function_name = config["name"]
            rust_name = config.get("rust_name", function_name)
            rust_extra = config.get("rust_extra", "")
            lhs = config.get("lhs", None)

            if "no_bare" not in config:
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
                if "rust_array_only" not in config:
                    if "free_only" not in config:
                        f.write(
                            test_array_function1(
                                rust_name,
                                function_name,
                                "axis=-1",
                                rust_name=rust_name,
                                # grab the rust_extra from the config otherwise use the default
                                rust_extra= config["axis"].get("rust_extra", "-1, None"),
                                lhs=lhs,
                            )
                        )
                    if "array_only" not in config:
                        f.write(
                            test_free_function1(
                                rust_name,
                                function_name,
                                "axis=-1",
                                rust_name=rust_name,
                                rust_extra= config["axis"].get("rust_extra", "-1, None"),
                                lhs=lhs,
                            )
                        )
                else:
                    f.write(
                        test_free_function1(
                            rust_name,
                            function_name,
                            "axis=-1",
                            rust_name=rust_name,
                            rust_extra= config["axis"].get("rust_extra", "-1, None"),
                            lhs=lhs,
                            via_rust_array=True
                        )
                    )

            if "axes" in config:
                if "rust_array_only" not in config:
                    if "free_only" not in config:
                        f.write(
                            test_array_function1(
                                rust_name,
                                function_name,
                                "axis=[0, -1]",
                                rust_name=rust_name,
                                rust_extra= config["axes"].get("rust_extra", "-1, None"),
                                lhs=(2, 3, 4, 3),
                            )
                        )
                    if "array_only" not in config:
                        f.write(
                            test_free_function1(
                                rust_name,
                                function_name,
                                "axis=[0, -1]",
                                rust_name=rust_name,
                                rust_extra= config["axes"].get("rust_extra", "-1, None"),
                                lhs=(2, 3, 4, 3),
                            )
                        )
                else:
                    f.write(
                        test_free_function1(
                            rust_name,
                            function_name,
                            "axis=[0, -1]",
                            rust_name=rust_name,
                            rust_extra= config["axes"].get("rust_extra", "-1, None"),
                            lhs=(2, 3, 4, 3),
                            via_rust_array=True
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

        # FFTs
        fft_functions = [
            ("fft", (100, 100), [dict(n=80), dict(n=120), dict(axis=0)]),
            ("ifft", (100,), [dict(n=80), dict(n=120), dict(axis=0)]),
            ("rfft", (100,), [dict(n=80), dict(n=120), dict(axis=0)]),
            ("irfft", (100,), [dict(n=80), dict(n=120), dict(axis=0)]),
            (
                "fft2",
                (8, 8, 8),
                [dict(s=[3, 4]), dict(axes=[0, 2]), dict(s=[10, 5], axes=[2, 1])],
            ),
            (
                "ifft2",
                (8, 8, 8),
                [dict(s=[3, 4]), dict(axes=[0, 2]), dict(s=[10, 5], axes=[2, 1])],
            ),
            (
                "fftn",
                (8, 8, 8),
                [dict(s=[3, 4]), dict(axes=[0, 2]), dict(s=[10, 5], axes=[2, 1])],
            ),
            (
                "ifftn",
                (8, 8, 8),
                [dict(s=[3, 4]), dict(axes=[0, 2]), dict(s=[10, 5], axes=[2, 1])],
            ),
            (
                "rfft2",
                (8, 8, 8),
                [dict(s=[3, 4]), dict(axes=[0, 2]), dict(s=[10, 5], axes=[2, 1])],
            ),
            (
                "irfft2",
                (8, 8, 8),
                [dict(s=[3, 4]), dict(axes=[0, 2]), dict(s=[10, 5], axes=[2, 1])],
            ),
            (
                "rfftn",
                (8, 8, 8),
                [dict(s=[3, 4]), dict(axes=[0, 2]), dict(s=[10, 5], axes=[2, 1])],
            ),
            (
                "irfftn",
                (8, 8, 8),
                [dict(s=[3, 4]), dict(axes=[0, 2]), dict(s=[10, 5], axes=[2, 1])],
            ),
        ]

        for fft, shape, args_array in fft_functions:
            f.write(test_fft(fft, value=shape))

            for args in args_array:
                f.write(test_fft(fft, value=shape, **args))

        # TODO: Test optimizers

        # TODO: Test layers

    print("Integration tests generated successfully")


if __name__ == "__main__":
    generate_integration_tests()
    subprocess.run(["cargo", "fmt", "--", "tests/integration_test.rs"], cwd="../")
