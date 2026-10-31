"""Differential tests for coverage batch W13 (Import/Export, serialisation, compression, cryptography).

Reference values:

* WXF (BinarySerialize): the byte sequences documented on reference.wolfram.com/language/ref/BinarySerialize.html
  and the WXF format specification (tutorial/WXFFormatDescription);
* JSON/CSV/TSV/XML text: the documented Export/ImportString results (tab-indented JSON, comma-separated rows,
  XMLObject["Document"] trees);
* cryptographic primitives: published test vectors - FIPS-197 (AES), NIST SP 800-38A (block modes), Schneier's
  Blowfish vectors, RC4 ("Key"/"Plaintext"), RFC 7914 (scrypt), RFC 6979 A.2.5 (deterministic ECDSA on P-256),
  RFC 8032 section 7.1 (Ed25519), BIP-340 (Schnorr) and the Keccak-256 / RIPEMD-160 reference digests;
* PEM keys and X.509 certificates generated with OpenSSL 3.5 (embedded below), including an OpenSSL ECDSA
  signature that must verify.
"""
import base64
import os
import tempfile

import pytest

from _harness import Pred, Raises, check, ev, ids
from mikoshilang.expr import Expr

D = tempfile.mkdtemp(prefix="w13_").replace("\\", "/")
os.environ["MIKOSHILANG_CACHE_BASE"] = D + "/cache"
os.environ["MIKOSHILANG_LOCAL_BASE"] = D + "/objects"

RSA_PKCS8 = """-----BEGIN PRIVATE KEY-----
MIICdwIBADANBgkqhkiG9w0BAQEFAASCAmEwggJdAgEAAoGBAMqW2ieSXITlao6A
RHhrdcTiF6Yki4hTMAZGa4WftkadjT0MfRYHcM8QccWvFNXBJ7EiuJO8LGI0WIab
pVdJsUp6PIk1HpgDTGkErJ2kONY3gkxj3ZPAhS4UdsQNd6/bUeDQg0ijG6xSMj5J
Zd4Z6FCiy3861uyWz40vKb0hA00HAgMBAAECgYA9rETHTTArKOgSb4vomNYmVKnc
Yz2zB5gUIPX0nNVRk/7+OcIOpb862ge4pX8TP5UEwMwt9VgRoVvdxp5D/2Qf+/Ov
yf+UdDGbizsgEfQ1rJIOz3LA3JqQ24TO5OF+BIyBJwq4rMvijdMydeUlcY/llfh9
MvCRvAIxA+To2vikkQJBAPk4j610OcLnYjC8dmjv3Pw8xKqolMXh3ynHHCKytJPT
6Qoc4mL4bKdjH3SsrL3ndVUGl/XMkt/n5/a0NM+JaAkCQQDQGZLHo2hg4g8/VCHG
9uiQ+ejdILfmsUAH/A/UfTd1SyjCJ4dtDalm6IXkUZMZpcpApEIPNyzCL0Upznu0
3LCPAkEA8yTmsEB7dhcOqiPEln6mKAUVjrZoZ2MFuZnDbmrRWIRfkPk7OXnLb8d6
UfeBlMkzpN+I9EtBZGe53OxVtcEGGQJBALTsJvKunQbuQzdZ2+vxqoyw2cW2tmw4
2mSWvuSM4PR3Hb5/Q3yKjidMLWwth+iafVbp+AzmeWFyW31XabH61P0CQCTyqpWO
JqCIqluhDCOjp9nCzZqeHcCZJ2FPWEXsREPgnmXAQYRgoUzWDGVAAY1SYt3ZhCTA
P0JoKseP+ZscxbM=
-----END PRIVATE KEY-----
"""
RSA_CERT = """-----BEGIN CERTIFICATE-----
MIICODCCAaGgAwIBAgICEjQwDQYJKoZIhvcNAQELBQAwNjERMA8GA1UEAwwIVzEz
IFRlc3QxFDASBgNVBAoMC01pa29zaGlMYW5nMQswCQYDVQQGEwJHQjAgFw0yNjEw
MzExODQ2MzFaGA8yMTI2MTAwNzE4NDYzMVowNjERMA8GA1UEAwwIVzEzIFRlc3Qx
FDASBgNVBAoMC01pa29zaGlMYW5nMQswCQYDVQQGEwJHQjCBnzANBgkqhkiG9w0B
AQEFAAOBjQAwgYkCgYEAypbaJ5JchOVqjoBEeGt1xOIXpiSLiFMwBkZrhZ+2Rp2N
PQx9FgdwzxBxxa8U1cEnsSK4k7wsYjRYhpulV0mxSno8iTUemANMaQSsnaQ41jeC
TGPdk8CFLhR2xA13r9tR4NCDSKMbrFIyPkll3hnoUKLLfzrW7JbPjS8pvSEDTQcC
AwEAAaNTMFEwHQYDVR0OBBYEFMBhRZtdsD6gdx9lop8q+xELTTZhMB8GA1UdIwQY
MBaAFMBhRZtdsD6gdx9lop8q+xELTTZhMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZI
hvcNAQELBQADgYEANgTrANcHtXRQR4N4KgqJXttzJhyJGc9FclJdBJRieUdvScxf
p+Wsp5smqCrJCjcGNmWybJ3cfkoN3snx6WVt+LBfPSxMVpNIjoxHS3R0sHc7aBjF
3h5aTkN3wqzq5k5RxifqrgGDNGnmbZ7Kr6+Een40Gbx+X/6JHUeWuyPYZrY=
-----END CERTIFICATE-----
"""
EC_SEC1 = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIH1Oe4rRvKPzy02Y/OWbVHNxu7xYmNw7g8F+i1XKZ47zoAoGCCqGSM49
AwEHoUQDQgAE5aG/2ue5pfQ/qK3QbVC6XuKgbPuMpty+/Wdj0qgBactKxMa4hJoU
P4102TPJabpOA++12iqyKHT/kt64xXu1Ug==
-----END EC PRIVATE KEY-----
"""
EC_CERT = """-----BEGIN CERTIFICATE-----
MIIBZjCCAQygAwIBAgIBBzAKBggqhkjOPQQDAzARMQ8wDQYDVQQDDAZXMTMgRUMw
IBcNMjYxMDMxMTg0NjMxWhgPMjEyNjEwMDcxODQ2MzFaMBExDzANBgNVBAMMBlcx
MyBFQzBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABOWhv9rnuaX0P6it0G1Qul7i
oGz7jKbcvv1nY9KoAWnLSsTGuISaFD+NdNkzyWm6TgPvtdoqsih0/5LeuMV7tVKj
UzBRMB0GA1UdDgQWBBTKcuL/nC4Y+/v4WX4KCUpVMOSjJDAfBgNVHSMEGDAWgBTK
cuL/nC4Y+/v4WX4KCUpVMOSjJDAPBgNVHRMBAf8EBTADAQH/MAoGCCqGSM49BAMD
A0gAMEUCIQCnqV5l8M6GiL09fFc47BFy0h/r7s0NZiNYl9+WJESVhQIgX9qXqtH5
aR4I06drbh1X5X5lhw9XlvqiBuC97PpcnmQ=
-----END CERTIFICATE-----
"""
EC_SPKI = """-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE5aG/2ue5pfQ/qK3QbVC6XuKgbPuM
pty+/Wdj0qgBactKxMa4hJoUP4102TPJabpOA++12iqyKHT/kt64xXu1Ug==
-----END PUBLIC KEY-----
"""


for _n, _t in (("rsa8.pem", RSA_PKCS8), ("rsacert.pem", RSA_CERT), ("ec1.pem", EC_SEC1), ("eccert.pem", EC_CERT),
               ("ecpub.pem", EC_SPKI)):
    with open(os.path.join(D, _n), "w") as _fh:
        _fh.write(_t)
with open(os.path.join(D, "signed.txt"), "wb") as _fh:
    _fh.write(b"w13 signed file")
with open(os.path.join(D, "data.csv"), "w") as _fh:
    _fh.write("1,2\n3,4\n")
with open(os.path.join(D, "page.txt"), "w") as _fh:
    _fh.write("cached page")

# an OpenSSL `dgst -sha256 -sign ec1.pem signed.txt` signature (DER), split into r and s
SIG_R = "93257697a744e0ac937a63ba6ae00f278a9c1685b2b8ab2b29673dcb6e49802a"
SIG_S = "2382b13ea0c62c9ea620d6209f94e200772d0f331b2646b03529313b03c1ccf0"


def T(src, text):
    return (src, Pred(lambda g, t=text: g == t, repr(text)))


def BA(src, hexstr):
    """src evaluates to a ByteArray with exactly these bytes."""
    want = bytes.fromhex(hexstr)

    def f(g):
        return isinstance(g, Expr) and g.head == "ByteArray" and base64.b64decode(g.args[0]) == want
    return (src, Pred(f, "ByteArray " + hexstr[:40]))


def HEAD(src, head):
    return (src, Pred(lambda g, h=head: isinstance(g, Expr) and g.head == h, "head " + head))


def P(src, f, what):
    return (src, Pred(f, what))


def hexs(s):
    return f'ByteArray["{base64.b64encode(bytes.fromhex(s)).decode()}"]'


def F(name):
    return f'"{D}/{name}"'


AES_PT = hexs("00112233445566778899aabbccddeeff")
SP_KEY, SP_PT = hexs("2b7e151628aed2a6abf7158809cf4f3c"), hexs("6bc1bee22e409f96e93d7e117393172a")
SP_IV = hexs("000102030405060708090a0b0c0d0e0f")


def ENC(keyexpr, pt, method=""):
    return f'Encrypt[{keyexpr}, {pt}{method}]["Data"]'


SERIALISATION = [
    BA("BinarySerialize[{1, 2, 3}]", "383a66037304"+"4c697374"+"430143024303"),
    BA("BinarySerialize[{1, -1, ByteArray[{1, 2, 3}]}]", "383a6603730" + "44c69737443014" + "3ff4203010203"),
    BA("BinarySerialize[\"abc\"]", "383a5303616263"),
    BA("BinarySerialize[w13sym]", "383a730d476c6f62616c6077313373796d"),
    ("BinaryDeserialize[BinarySerialize[w13f[x, \"y\", 2.5, 3/4]]]", "w13f[x, \"y\", 2.5, 3/4]"),
    ("BinaryDeserialize[BinarySerialize[<|\"a\" -> 1, \"b\" :> w13q|>]]", "<|\"a\" -> 1, \"b\" :> w13q|>"),
    ("BinaryDeserialize[BinarySerialize[Range[50], PerformanceGoal -> \"Size\"]]", "Range[50]"),
    P("BinarySerialize[Range[50], PerformanceGoal -> \"Size\"]",
      lambda g: base64.b64decode(g.args[0])[:3] == b"8C:", "8C: header"),
    ("BinaryDeserialize[ByteArray[{56, 58, 67, 7}]]", "7"),
    ("BinaryDeserialize[BinarySerialize[{1, 2}], w13h]", "w13h[{1, 2}]"),
    ("BinaryDeserialize[ByteArray[{1, 2, 3}]]", Raises(ValueError)),
    ("Uncompress[Compress[{w13c, 1.5, \"s\", 2^100}]]", "{w13c, 1.5, \"s\", 2^100}"),
    P("Compress[x^2]", lambda g: isinstance(g, str) and g.startswith("1:"), "1: prefix"),
    ("Uncompress[Compress[N[Pi, 30]]] === N[Pi, 30]", "True"),
    ("Uncompress[Compress[1 + 2 I]]", "1 + 2 I"),
    ("Uncompress[Compress[{1, 2}], w13hold]", "w13hold[{1, 2}]"),
    ("Uncompress[\"not compressed\"]", Raises(ValueError)),
    ("Compress[x, y, z]", Raises(NotImplementedError)),
]

FORMATS = [
    T("ExportString[{1, 2, 3}, \"JSON\"]", "[\n\t1,\n\t2,\n\t3\n]"),
    T("ExportString[<|\"a\" -> 1|>, \"RawJSON\"]", "{\n\t\"a\": 1\n}"),
    T("ExportString[{{1, 2}, {3, 4}}, \"CSV\"]", "1,2\n3,4"),
    T("ExportString[{{\"a\", \"b c\"}}, \"TSV\"]", "a\tb c"),
    T("ExportString[XMLElement[\"a\", {\"k\" -> \"v\"}, {\"t\"}], \"XML\"]", "<a k=\"v\">t</a>"),
    T("ExportString[\"hi\", \"Base64\"]", "aGk="),
    ("ExportString[x, \"NoSuchFormat\"]", Raises(ValueError)),
    ("ImportString[\"1,2\\n3,4\", \"CSV\"]", "{{1, 2}, {3, 4}}"),
    ("ImportString[\"a\\tb\\n1\\t2.5\", \"TSV\"]", "{{\"a\", \"b\"}, {1, 2.5}}"),
    ("ImportString[\"[1, 2, {\\\"a\\\": true}]\", \"JSON\"]", "{1, 2, {\"a\" -> True}}"),
    ("ImportString[\"{\\\"a\\\": [1, null]}\", \"RawJSON\"]", "<|\"a\" -> {1, Null}|>"),
    ("ImportString[\"<a x=\\\"1\\\">hi</a>\", \"XML\"]",
     "XMLObject[\"Document\"][{}, XMLElement[\"a\", {\"x\" -> \"1\"}, {\"hi\"}], {}]"),
    ("ImportString[\"1,2\\n3,4\", {\"CSV\", \"Dimensions\"}]", "{2, 2}"),
    ("ImportString[\"aGk=\", \"Base64\"]", "\"hi\""),
    ("ImportString[\"1,2\", \"NoSuchFormat\"]", Raises(ValueError)),
    BA("ExportByteArray[{1, 2}, \"JSON\"]", "5b0a09312c0a09320a5d"),
    BA("ExportByteArray[{1, 2, 255}, \"Byte\"]", "0102ff"),
    BA("ExportByteArray[{1, 256}, \"UnsignedInteger16\"]", "01000001"),
    ("ExportByteArray[x, \"NoSuchFormat\"]", Raises(ValueError)),
    ("ImportByteArray[ByteArray[{91, 49, 93}], \"JSON\"]", "{1}"),
    ("ImportByteArray[ByteArray[{1, 0, 0, 1}], \"UnsignedInteger16\"]", "{1, 256}"),
    ("ImportByteArray[ExportByteArray[{1.5, -2.}, \"Real64\"], \"Real64\"]", "{1.5, -2.}"),
    ("ImportByteArray[{1, 2}, \"JSON\"]", Raises(ValueError)),
    (f"Export[{F('e1.csv')}, {{{{1, 2}}, {{3, 4}}}}]", f"{F('e1.csv')}"),
    (f"Export[{F('e2.json')}, {{1, 2}}]; Import[{F('e2.json')}]", "{1, 2}"),
    (f"Export[{F('e3.dat')}, \"x\", \"Text\"]; Import[{F('e3.dat')}, \"Text\"]", "\"x\""),
    (f"Export[{F('e4.wxf')}, w13e[1]]; Import[{F('e4.wxf')}]", "w13e[1]"),
    (f"Export[{F('e5.csv.gz')}, {{{{1, 2}}}}]; Import[{F('e5.csv.gz')}]", "{{1, 2}}"),
    (f"Export[{F('e6.zip')}, {{\"a.json\" -> {{1}}, \"b.txt\" -> \"t\"}}]; Import[{F('e6.zip')}, \"FileNames\"]",
     "{\"a.json\", \"b.txt\"}"),
    (f"Export[{F('e7.unknownext')}, 1]", Raises(ValueError)),
    (f"Export[{F('e8.csv')}, {{{{1}}}}, MetaInformation -> {{\"a\" -> 1}}]", Raises(NotImplementedError)),
    (f"Import[{F('data.csv')}]", "{{1, 2}, {3, 4}}"),
    (f"Import[{F('data.csv')}, \"RowCount\"]", "2"),
    (f"Import[{F('data.csv')}, {{\"CSV\", \"Data\", 2}}]", "{3, 4}"),
    (f"Import[{F('data.csv')}, \"Elements\"]", "{\"ColumnCount\", \"Data\", \"Dimensions\", \"Grid\", \"RawData\", "
                                               "\"RowCount\"}"),
    (f"Import[{F('missing.csv')}]", Raises(ValueError)),
    (f"Import[File[{F('data.csv')}]]", "{{1, 2}, {3, 4}}"),
    (f"Import[{F('data.csv')}, \"CSV\", CharacterEncoding -> \"UTF8\"]", "{{1, 2}, {3, 4}}"),
    (f"Import[URL[\"file://{D}/data.csv\"]]", "{{1, 2}, {3, 4}}"),
    (f"Import[RemoteFile[URL[\"file://{D}/data.csv\"]], \"RowCount\"]", "2"),
    (f"FileConvert[{F('data.csv')} -> \"JSON\"]", f"{F('data.json')}"),
    (f"FileConvert[{F('data.csv')} -> {F('conv.tsv')}]; Import[{F('conv.tsv')}]", "{{1, 2}, {3, 4}}"),
    (f"FileConvert[{F('nonexistent.csv')} -> \"JSON\"]", Raises(ValueError)),
]

REGISTRY = [
    P("$ImportFormats", lambda g: {"CSV", "JSON", "XML", "PEM", "WXF"} <= set(g.args), "contains CSV, JSON, ..."),
    P("$ExportFormats", lambda g: {"CSV", "JSON", "XML", "PEM", "MX"} <= set(g.args), "contains CSV, JSON, ..."),
    P("Length[$ImportFormats] > 20", lambda g: g is True, "True"),
    P("MemberQ[$ExportFormats, \"Base64\"]", lambda g: g is True, "True"),
    ("MemberQ[$ImportFormats, \"NoSuchFormat\"]", "False"),
    ("Complement[{\"CSV\", \"TSV\"}, $ImportFormats, $ExportFormats]", "{}"),
    ("FileFormatProperties[\"CSV\", \"MIMETypes\"]", "{\"text/csv\", \"text/comma-separated-values\"}"),
    ("FileFormatProperties[\"CSV\", \"FileNamePatterns\"]", "{\"*.csv\"}"),
    ("FileFormatProperties[\"PEM\", \"DefaultImportElement\"]", "\"Data\""),
    ("FileFormatProperties[\"NoSuchFormat\"]", Raises(ValueError)),
    ("SetFileFormatProperties[\"TSV\", \"DefaultImportElement\" -> \"Data\"]", "Null"),
    ("SetFileFormatProperties[\"TSV\", \"MIMETypes\" -> {\"text/tab-separated-values\"}]; "
     "FileFormatProperties[\"TSV\", \"MIMETypes\"]", "{\"text/tab-separated-values\"}"),
    ("SetFileFormatProperties[\"NoSuchFormat\", \"MIMETypes\" -> {}]", Raises(ValueError)),
    ("FileNameToFormatList[\"a.pem\"]", "{\"PEM\"}"),
    ("FileNameToFormatList[\"x.csv\"]", "{\"CSV\"}"),
    ("FileNameToFormatList[\"x.tsv\"]", "{\"TSV\"}"),
    ("MIMETypeToFormatList[\"text/csv\"]", "{\"CSV\"}"),
    ("MIMETypeToFormatList[\"application/x-pem-file\"]", "{\"PEM\"}"),
    ("MIMETypeToFormatList[\"application/no-such\"]", "{}"),
    ("FileFormat[\"x.json\"]", "\"JSON\""),
    (f"FileFormat[{F('data.csv')}]", "\"CSV\""),
    (f"FileFormat[{F('rsacert.pem')}]", "\"PEM\""),
    (f"FileFormatQ[{F('data.csv')}, \"CSV\"]", "True"),
    (f"FileFormatQ[{F('data.csv')}, \"JSON\"]", "False"),
    ("FileFormatQ[\"/no/such/file.csv\", \"CSV\"]", "False"),
    ("DefaultElement", "DefaultElement"), ("ImportOptions", "ImportOptions"), ("ConversionRules", "ConversionRules"),
    ("IncludeMetaInformation", "IncludeMetaInformation"), ("MetaInformation", "MetaInformation"),
    ("{DefaultElement -> \"Data\"}", "{DefaultElement -> \"Data\"}"),
    ("{ImportOptions -> {}}", "{ImportOptions -> {}}"),
    ("{ConversionRules -> {}}", "{ConversionRules -> {}}"),
    ("{IncludeMetaInformation -> All}", "{IncludeMetaInformation -> All}"),
    (f"Import[{F('data.csv')}, ConversionRules -> {{}}]", Raises(ValueError)),
    (f"Export[{F('e9.csv')}, {{{{1}}}}, IncludeMetaInformation -> All]", Raises(ValueError)),
    ("Head[DefaultElement]", "Symbol"),
    ("Head[ImportOptions]", "Symbol"),
]

FILES = [
    (f"w13put = {{1, 2}}; Put[w13put, {F('p1.wl')}]; Get[{F('p1.wl')}]", "{1, 2}"),
    (f"Put[1, w13pv, {F('p2.wl')}]; Import[{F('p2.wl')}, \"Text\"]", "\"1\\nw13pv\""),
    (f"Put[{F('p3.wl')}]; Import[{F('p3.wl')}, \"Text\"]", "\"\""),
    (f"{{7, 8}} >> {F('p4.wl')}; Get[{F('p4.wl')}]", "{7, 8}"),
    (f"Put[1, 2]", Raises(TypeError)),
    (f"Put[1, {F('p5.wl')}]; PutAppend[w13pa^2, {F('p5.wl')}]; Import[{F('p5.wl')}, \"Text\"]", "\"1\\nw13pa^2\""),
    (f"PutAppend[3, {F('p6.wl')}]; PutAppend[4, {F('p6.wl')}]; Get[{F('p6.wl')}]", "4"),
    (f"5 >>> {F('p7.wl')}; Get[{F('p7.wl')}]", "5"),
    ("PutAppend[1, 2]", Raises(TypeError)),
    (f"w13sv = 5; Save[{F('s1.wl')}, w13sv]; Import[{F('s1.wl')}, \"Text\"]", "\"w13sv = 5\\n \""),
    (f"w13sf[x_] := x^2; Save[{F('s2.wl')}, w13sf]; Clear[w13sf]; Get[{F('s2.wl')}]; w13sf[3]", "9"),
    (f"w13sa = 1; w13sb = 2; Save[{F('s3.wl')}, {{w13sa, w13sb}}]; w13sa = 0; w13sb = 0; Get[{F('s3.wl')}]; "
     f"{{w13sa, w13sb}}", "{1, 2}"),
    ("Save[1, w13sv]", Raises(TypeError)),
    (f"w13dv = {{1, 2}}; DumpSave[{F('d1.mx')}, w13dv]", "{{1, 2}}"),
    (f"w13dw = 7; DumpSave[{F('d2.mx')}, w13dw]; Clear[w13dw]; Get[{F('d2.mx')}]; w13dw", "7"),
    (f"w13dg[n_] := n + 1; DumpSave[{F('d3.mx')}, w13dg]; Clear[w13dg]; Get[{F('d3.mx')}]; w13dg[1]", "2"),
    ("DumpSave[1, w13dv]", Raises(TypeError)),
    (f"CreateArchive[{F('data.csv')}, {F('a1.zip')}]", f"{F('a1.zip')}"),
    (f"CreateArchive[{F('data.csv')}, {F('a2.tar.gz')}]; ExtractArchive[{F('a2.tar.gz')}, {F('x2')}]",
     f"{{{F('x2/data.csv')}}}"),
    (f"CreateArchive[{F('data.csv')}, {F('a1.zip')}]", Raises(ValueError)),
    (f"CreateArchive[{F('data.csv')}, {F('a3.zip')}]; ExtractArchive[{F('a3.zip')}, {F('x3')}]; "
     f"Import[{F('x3/data.csv')}]", "{{1, 2}, {3, 4}}"),
    (f"ExtractArchive[{F('data.csv')}, {F('x4')}]", Raises(ValueError)),
    (f"CreateArchive[{F('data.csv')}, {F('a5.tar')}]; Length[ExtractArchive[{F('a5.tar')}, {F('x5')}]]", "1"),
]

MISC = [
    P("CreateUUID[]", lambda g: isinstance(g, str) and len(g) == 36 and g[14] == "4", "UUID v4 string"),
    P("CreateUUID[\"w13-\"]", lambda g: g.startswith("w13-") and len(g) == 40, "prefixed UUID"),
    ("CreateUUID[] === CreateUUID[]", "False"),
    ("CreateUUID[1]", Raises(TypeError)),
    ("URL[\"https://a.b/c\"]", "URL[\"https://a.b/c\"]"),
    ("URL[\"https://a.b/c\"] // Head", "URL"),
    ("URL[5]", Raises(TypeError)),
    ("RemoteFile[\"https://x.y/z\"]", "RemoteFile[URL[\"https://x.y/z\"]]"),
    ("RemoteFile[URL[\"https://x.y/z\"]]", "RemoteFile[URL[\"https://x.y/z\"]]"),
    ("RemoteFile[3]", Raises(TypeError)),
    T("$CacheBaseDirectory", D + "/cache"),
    T("$DefaultLocalBase", D + "/objects"),
    T("$LocalBase", D + "/objects"),
    P("$LocalSymbolBase", lambda g: isinstance(g, str) and len(g) > 0, "a directory name"),
    ("StringQ[$CacheBaseDirectory]", "True"), ("StringQ[$LocalSymbolBase]", "True"),
    (f"StringStartsQ[LocalCache[URL[\"file://{D}/page.txt\"]][[1]], \"file://\" <> $CacheBaseDirectory]", "True"),
    ("$LocalBase === $DefaultLocalBase", "True"),
    ("$LocalSymbolBase === $LocalSymbolBase", "True"),
    ("StringQ[$DefaultLocalBase]", "True"), ("StringQ[$LocalBase]", "True"),
    P(f"LocalCache[URL[\"file://{D}/page.txt\"]]",
      lambda g: g.head == "LocalObject" and g.args[0].startswith("file://" + D + "/cache/"), "cached object"),
    (f"Import[LocalCache[URL[\"file://{D}/page.txt\"]], \"Text\"]", "\"cached page\""),
    ("LocalCache[5]", Raises(TypeError)),
    ("LocalCache[CloudObject[\"x\"]]", Raises(NotImplementedError)),
    ("PersistenceLocation[\"KernelSession\"]", "PersistenceLocation[\"KernelSession\"]"),
    P("PersistenceLocation[\"Local\"]", lambda g: g.head == "PersistenceLocation" and len(g.args) == 2,
      "with a LocalObject"),
    ("PersistenceLocation[\"Bogus\"]", Raises(ValueError)),
    ("ToMemory[{1, 2}]", "{1, 2}"),
    ("ToMemory[w13tm]", "w13tm"),
    ("ToMemory[]", Raises(ValueError)),
    ("CachePersistence", "CachePersistence"), ("PersistenceTime", "PersistenceTime"),
    ("VerifySecurityCertificates", "VerifySecurityCertificates"),
    ("{CachePersistence -> Infinity}", "{CachePersistence -> Infinity}"),
    ("{PersistenceTime -> 10}", "{PersistenceTime -> 10}"), ("Head[PersistenceTime]", "Symbol"),
    ("{VerifySecurityCertificates -> False}", "{VerifySecurityCertificates -> False}"),
    ("Head[CachePersistence]", "Symbol"),
    ("Head[VerifySecurityCertificates]", "Symbol"),
    ("Boxes", "Boxes"), ("{Boxes -> True}", "{Boxes -> True}"), ("Head[Boxes]", "Symbol"),
    ("XMLElement[\"a\", {}, {\"b\"}]", "XMLElement[\"a\", {}, {\"b\"}]"),
    ("ImportString[\"<r><c/></r>\", {\"XML\", \"XMLElement\"}]", "XMLElement[\"r\", {}, {XMLElement[\"c\", {}, {}]}]"),
    ("Head[XMLElement[\"a\", {}, {}]]", "XMLElement"),
    ("Head[ImportString[\"<a/>\", \"XML\"]]", "XMLObject[\"Document\"]"),
    ("Head[ImportString[\"<a/>\", \"XML\"]] === XMLObject[\"Document\"]", "True"),
    ("XMLObject[\"Document\"][{}, XMLElement[\"a\", {}, {}], {}]",
     "XMLObject[\"Document\"][{}, XMLElement[\"a\", {}, {}], {}]"),
    T("ExportString[XMLObject[\"Document\"][{}, XMLElement[\"a\", {}, {}], {}], \"XML\"]",
      "<a/>"),
    ("ImportedObject[<|\"OriginalFormat\" -> \"PEM\", \"RawContent\" -> \"x\"|>][\"OriginalFormat\"]", "\"PEM\""),
    ("ImportedObject[<|\"OriginalFormat\" -> \"PEM\", \"RawContent\" -> \"x\"|>][\"RawContent\"]", "\"x\""),
    ("ImportedObject[<|\"OriginalFormat\" -> \"PEM\"|>][\"Nope\"]", "Missing[\"NotAvailable\"]"),
    T("TextString[<|\"a\" -> 1, \"b\" -> 2|>, AssociationFormat -> {\"[\", \"; \", \"]\", \"=\"}]", "[a=1; b=2]"),
    T("TextString[<|\"a\" -> 1, w13af -> {1, 2}|>]", "<|a -> 1, w13af -> {1, 2}|>"),
    T("TextString[{<|1 -> 2|>}, AssociationFormat -> {\"(\", \",\", \")\"}]", "{(1 -> 2)}"),
    T("TextString[<|x -> 1, y -> 2|>, AssociationFormat -> \" & \"]", "x -> 1 & y -> 2"),
    ("TextString[<|1 -> 2|>, AssociationFormat -> 5]", Raises(ValueError)),
]

HASHES_AND_CIPHERS = [
    # FIPS-197 appendix C
    BA(ENC(f"SymmetricKey[<|\"Cipher\" -> \"AES128\", \"Key\" -> {hexs('000102030405060708090a0b0c0d0e0f')}|>]",
           AES_PT, ", Method -> <|\"BlockMode\" -> \"ECB\", \"Padding\" -> None|>"),
       "69c4e0d86a7b0430d8cdb78070b4c55a"),
    BA(ENC(f"SymmetricKey[<|\"Cipher\" -> \"AES192\", \"BlockMode\" -> \"ECB\", \"Key\" -> "
           f"{hexs('000102030405060708090a0b0c0d0e0f1011121314151617')}|>]", AES_PT,
           ", Method -> <|\"Padding\" -> None|>"), "dda97ca4864cdfe06eaf70a0ec0d7191"),
    BA(ENC(f"SymmetricKey[<|\"Cipher\" -> \"AES256\", \"BlockMode\" -> \"ECB\", \"Key\" -> "
           f"{hexs(bytes(range(32)).hex())}|>]", AES_PT, ", Method -> <|\"Padding\" -> None|>"),
       "8ea2b7ca516745bfeafc49904b496089"),
    # SP 800-38A F.2.1, F.3.13, F.4.1, F.5.1
    BA(ENC(f"SymmetricKey[<|\"Cipher\" -> \"AES128\", \"BlockMode\" -> \"CBC\", \"Key\" -> {SP_KEY}, "
           f"\"InitializationVector\" -> {SP_IV}|>]", SP_PT, ", Method -> <|\"Padding\" -> None|>"),
       "7649abac8119b246cee98e9b12e9197d"),
    BA(ENC(f"SymmetricKey[<|\"Cipher\" -> \"AES128\", \"BlockMode\" -> \"CFB\", \"Key\" -> {SP_KEY}, "
           f"\"InitializationVector\" -> {SP_IV}|>]", SP_PT), "3b3fd92eb72dad20333449f8e83cfb4a"),
    BA(ENC(f"SymmetricKey[<|\"Cipher\" -> \"AES128\", \"BlockMode\" -> \"OFB\", \"Key\" -> {SP_KEY}, "
           f"\"InitializationVector\" -> {SP_IV}|>]", SP_PT), "3b3fd92eb72dad20333449f8e83cfb4a"),
    BA(ENC(f"SymmetricKey[<|\"Cipher\" -> \"AES128\", \"BlockMode\" -> \"CTR\", \"Key\" -> {SP_KEY}, "
           f"\"InitializationVector\" -> {hexs('f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff')}|>]", SP_PT),
       "874d6191b620e3261bef6864990db6ce"),
    # Blowfish (Schneier's vectors) and RC4
    BA(ENC(f"SymmetricKey[<|\"Cipher\" -> \"Blowfish\", \"BlockMode\" -> \"ECB\", \"Key\" -> {hexs('00' * 8)}|>]",
           hexs("00" * 8), ", Method -> <|\"Padding\" -> None|>"), "4ef997456198dd78"),
    BA(ENC(f"SymmetricKey[<|\"Cipher\" -> \"Blowfish\", \"BlockMode\" -> \"ECB\", \"Key\" -> {hexs('ff' * 8)}|>]",
           hexs("ff" * 8), ", Method -> <|\"Padding\" -> None|>"), "51866fd5b85ecb8a"),
    BA(ENC("SymmetricKey[<|\"Cipher\" -> \"RC4\", \"Key\" -> ByteArray[{75, 101, 121}]|>]", "\"Plaintext\""),
       "bbf316e8d940af0ad3"),
]

SYMMETRIC = [
    ("w13k1 = GenerateSymmetricKey[]; w13k1[\"Cipher\"]", "\"AES256\""),
    ("GenerateSymmetricKey[][\"BlockMode\"]", "\"CBC\""),
    P("GenerateSymmetricKey[Method -> \"AES128\"][\"Key\"]", lambda g: len(base64.b64decode(g.args[0])) == 16,
      "16 bytes"),
    ("GenerateSymmetricKey[\"w13pw\"] === GenerateSymmetricKey[\"w13pw\"]", "True"),
    ("GenerateSymmetricKey[\"w13pw\"] === GenerateSymmetricKey[\"w13pw2\"]", "False"),
    ("GenerateSymmetricKey[ByteArray[Range[16]], Method -> \"AES128\"][\"Key\"]", "ByteArray[\"AQIDBAUGBwgJCgsMDQ4PEA==\"]"),
    ("GenerateSymmetricKey[Method -> <|\"Cipher\" -> \"Blowfish\"|>][\"Cipher\"]", "\"Blowfish\""),
    ("GenerateSymmetricKey[Method -> \"DES\"]", Raises(NotImplementedError)),
    ("GenerateSymmetricKey[Method -> \"Nope\"]", Raises(ValueError)),
    ("GenerateSymmetricKey[ByteArray[{1, 2}], Method -> \"AES128\"]", Raises(ValueError)),
    ("SymmetricKey[<|\"Cipher\" -> \"AES128\", \"Key\" -> ByteArray[Range[16]]|>][\"Cipher\"]", "\"AES128\""),
    ("Head[SymmetricKey[<|\"Cipher\" -> \"AES128\", \"Key\" -> ByteArray[Range[16]]|>]]", "SymmetricKey"),
    ("SymmetricKey[<|\"Cipher\" -> \"AES128\"|>]", Raises(ValueError)),
    ("SymmetricKey[5]", Raises(TypeError)),
    ("SymmetricKey[<|\"Cipher\" -> \"AES128\", \"Key\" -> ByteArray[Range[16]]|>][\"Properties\"]",
     "{\"Cipher\", \"Key\"}"),
    ("w13k2 = GenerateSymmetricKey[]; Decrypt[w13k2, Encrypt[w13k2, \"hello\"]]", "\"hello\""),
    ("w13k3 = GenerateSymmetricKey[]; Decrypt[w13k3, Encrypt[w13k3, {1, w13x, 2.5}]]", "{1, w13x, 2.5}"),
    BA("Decrypt[\"w13pass\", Encrypt[\"w13pass\", ByteArray[{1, 2, 3}]]]", "010203"),
    ("w13k4 = GenerateSymmetricKey[Method -> \"Blowfish\"]; Decrypt[w13k4, Encrypt[w13k4, \"fish\"]]", "\"fish\""),
    ("w13k5 = GenerateSymmetricKey[Method -> \"RC4\"]; Decrypt[w13k5, Encrypt[w13k5, \"stream\"]]", "\"stream\""),
    ("w13k6 = GenerateSymmetricKey[Method -> <|\"BlockMode\" -> \"CTR\"|>]; Decrypt[w13k6, Encrypt[w13k6, \"ctr\"]]",
     "\"ctr\""),
    ("w13k7 = GenerateSymmetricKey[]; Decrypt[w13k7, Encrypt[w13k7, \"x\", Method -> <|\"BlockMode\" -> \"OFB\"|>]]",
     "\"x\""),
    ("w13k8 = GenerateSymmetricKey[]; Decrypt[w13k8][Encrypt[w13k8][\"op\"]]", "\"op\""),
    ("Encrypt[GenerateSymmetricKey[], \"t\"][\"OriginalForm\"]", "String"),
    ("Encrypt[GenerateSymmetricKey[], 1][\"OriginalForm\"]", "Expression"),
    HEAD("Encrypt[GenerateSymmetricKey[], \"t\"]", "EncryptedObject"),
    ("Encrypt[GenerateSymmetricKey[], \"t\"] === Encrypt[GenerateSymmetricKey[], \"t\"]", "False"),
    ("Decrypt[GenerateSymmetricKey[\"a\"], Encrypt[GenerateSymmetricKey[\"b\"], \"secret text\"]]", Raises(ValueError)),
    ("Encrypt[5, \"x\"]", Raises(TypeError)),
    ("Decrypt[GenerateSymmetricKey[], \"x\"]", Raises(TypeError)),
    ("w13eo = Encrypt[\"pw\", \"x\"]; EncryptedObject[w13eo[[1]]] === w13eo", "True"),
    ("Encrypt[\"pw\", \"x\"][\"Properties\"]", "{\"Data\", \"InitializationVector\", \"OriginalForm\"}"),
    ("EncryptedObject[<|\"x\" -> 1|>]", Raises(ValueError)),
    ("Head[Encrypt[\"pw\", 1]] === EncryptedObject", "True"),
    (f"w13fk = GenerateSymmetricKey[\"w13file\"]; EncryptFile[w13fk, {F('data.csv')}]", f"{F('data.csv.mx')}"),
    (f"Head[Import[{F('data.csv.mx')}]]", "EncryptedObject"),
    (f"EncryptFile[w13fk, {F('data.csv')}]", Raises(ValueError)),
    (f"EncryptFile[w13fk, {F('data.csv')}, {F('enc2.bin')}]; DecryptFile[w13fk, {F('enc2.bin')}, {F('dec2.csv')}]; "
     f"Import[{F('dec2.csv')}]", "{{1, 2}, {3, 4}}"),
    (f"DecryptFile[w13fk, {F('data.csv.mx')}, {F('dec1.csv')}]", f"{F('dec1.csv')}"),
    (f"DecryptFile[w13fk, {F('data.csv.mx')}]", Raises(ValueError)),
    (f"DecryptFile[GenerateSymmetricKey[\"wrong\"], {F('data.csv.mx')}, {F('dec3.csv')}]", Raises(ValueError)),
    (f"DecryptFile[w13fk, {F('data.csv.mx')}, {F('dec1.csv')}, OverwriteTarget -> True]", f"{F('dec1.csv')}"),
    (f"EncryptFile[w13fk, {F('nonexistent.txt')}]", Raises(ValueError)),
]

DERIVED = [
    BA("GenerateDerivedKey[\"password\", \"NaCl\", Method -> <|\"N\" -> 1024, \"r\" -> 8, \"p\" -> 16|>]"
       "[\"DerivedKey\"]",
       "fdbabe1c9d3472007856e7190d01e9fe7c6ad7cbc8237830e77376634b3731622eaf30d92e22a3886ff109279d9830dac727afb94a"
       "83ee6d8360cbdfa2cc0640"),
    ("GenerateDerivedKey[\"pw\", \"salt\", Method -> <|\"N\" -> 16, \"KeySize\" -> 16|>][\"FunctionParameters\"]",
     "<|\"KeySize\" -> 16, \"N\" -> 16, \"r\" -> 8, \"p\" -> 1|>"),
    ("GenerateDerivedKey[\"pw\"][\"Function\"]", "\"scrypt\""),
    P("GenerateDerivedKey[\"pw\"][\"Salt\"]", lambda g: len(base64.b64decode(g.args[0])) == 16, "16-byte salt"),
    T("GenerateDerivedKey[\"password\", \"NaCl\", Method -> <|\"N\" -> 1024, \"r\" -> 8, \"p\" -> 16, "
      "\"KeySize\" -> 8|>][\"PHCString\"]", "$scrypt$ln=10,r=8,p=16$TmFDbA$/bq+HJ00cgA"),
    ("GenerateDerivedKey[\"pw\", Method -> <|\"Function\" -> \"Argon2id\"|>]", Raises(NotImplementedError)),
    ("GenerateDerivedKey[\"pw\", Method -> <|\"N\" -> 1000|>]", Raises(ValueError)),
    ("GenerateDerivedKey[5]", Raises(TypeError)),
    ("w13dk = GenerateDerivedKey[\"pw\", \"s\", Method -> <|\"N\" -> 64|>]; DerivedKey[w13dk[\"PHCString\"]] === w13dk",
     "True"),
    ("DerivedKey[\"$scrypt$ln=10,r=8,p=16$TmFDbA$/bq+HJ00cgA\"][\"FunctionParameters\"][\"KeySize\"]", "8"),
    ("DerivedKey[\"not a phc string\"]", Raises(ValueError)),
    ("Head[GenerateDerivedKey[\"pw\", Method -> <|\"N\" -> 64|>]]", "DerivedKey"),
    ("w13dk2 = GenerateDerivedKey[\"right\", Method -> <|\"N\" -> 64|>]; VerifyDerivedKey[w13dk2, \"right\"]", "True"),
    ("w13dk3 = GenerateDerivedKey[\"right\", Method -> <|\"N\" -> 64|>]; VerifyDerivedKey[w13dk3, \"wrong\"]", "False"),
    ("VerifyDerivedKey[DerivedKey[\"$scrypt$ln=10,r=8,p=16$TmFDbA$/bq+HJ00cgA\"], \"password\"]", "True"),
    ("VerifyDerivedKey[5, \"x\"]", Raises(TypeError)),
    ("w13dk4 = GenerateDerivedKey[\"pw\", Method -> <|\"N\" -> 64, \"KeySize\" -> 32|>]; "
     "Decrypt[GenerateSymmetricKey[w13dk4], Encrypt[GenerateSymmetricKey[w13dk4], \"dk\"]]", "\"dk\""),
]

P256_X = "C9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721"
ED_SEED1 = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"
ED_SEED2 = "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb"
P256_KEY = (f"PrivateKey[<|\"Type\" -> \"EllipticCurve\", \"CurveName\" -> \"prime256v1\", "
            f"\"PrivateHexString\" -> \"{P256_X}\"|>]")

ASYMMETRIC = [
    ("w13kp = GenerateAsymmetricKeyPair[Method -> <|\"KeySize\" -> 1024|>]; w13kp[\"PublicKey\"][\"PublicKeySize\"]",
     "Quantity[1024, \"Bits\"]"),
    ("Keys[GenerateAsymmetricKeyPair[\"EdwardsCurve\"]]", "{\"PrivateKey\", \"PublicKey\"}"),
    ("GenerateAsymmetricKeyPair[\"EllipticCurve\"][\"PrivateKey\"][\"CurveName\"]", "\"secp256k1\""),
    ("GenerateAsymmetricKeyPair[Method -> <|\"Type\" -> \"EllipticCurve\", \"CurveName\" -> \"secp384r1\"|>]"
     "[\"PublicKey\"][\"CurveName\"]", "\"secp384r1\""),
    ("GenerateAsymmetricKeyPair[\"Bitcoin\"][\"PublicKey\"][\"Compressed\"]", "True"),
    ("StringLength[GenerateAsymmetricKeyPair[\"Ethereum\"][\"PublicKey\"][\"PublicHexString\"]]", "130"),
    ("GenerateAsymmetricKeyPair[\"DSA\"]", Raises(ValueError)),
    ("GenerateAsymmetricKeyPair[Method -> <|\"Type\" -> \"EdwardsCurve\", \"CurveName\" -> \"ed448\"|>]",
     Raises(NotImplementedError)),
    ("GenerateAsymmetricKeyPair[Method -> <|\"Type\" -> \"RSA\", \"KeySize\" -> 128|>]", Raises(ValueError)),
    # textbook RSA (n = 3233, e = 17, d = 2753)
    ("PrivateKey[<|\"Type\" -> \"RSA\", \"PublicModulus\" -> 3233, \"PublicExponent\" -> 17, "
     "\"PrivateExponent\" -> 2753|>][\"PrivateExponent\"]", "2753"),
    ("PrivateKey[<|\"Type\" -> \"RSA\", \"PublicModulus\" -> 3233, \"PublicExponent\" -> 17, "
     "\"PrivateExponent\" -> 2753|>][\"PublicKeySize\"]", "Quantity[12, \"Bits\"]"),
    ("PrivateKey[<|\"Type\" -> \"RSA\", \"PublicModulus\" -> 3233|>]", Raises(ValueError)),
    # secp256k1 with multiplier 3 (the BIP-340 vector 0 public key)
    T("PrivateKey[<|\"Type\" -> \"EllipticCurve\", \"CurveName\" -> \"secp256k1\", \"PrivateMultiplier\" -> 3|>]"
      "[\"PublicHexString\"]",
      "04f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9388f7b0f632de8140fe337e62a37f3566500a99934c"
      "2231b6cb9fd7584b8e672"),
    T(f"{P256_KEY}[\"PublicHexString\"]",
      "0460fed4ba255a9d31c961eb74c6356d68c049b8923b61fa6ce669622e60f29fb67903fe1008b8bc99a41ae9e95628bc64f2f1b20c2d7e"
      "9f5177a3c294d4462299"),
    ("PrivateKey[<|\"Type\" -> \"EllipticCurve\", \"CurveName\" -> \"secp256k1\", \"PrivateMultiplier\" -> 0|>]",
     Raises(ValueError)),
    ("PrivateKey[<|\"Type\" -> \"EllipticCurve\", \"CurveName\" -> \"nocurve\", \"PrivateMultiplier\" -> 2|>]",
     Raises(ValueError)),
    T(f"PrivateKey[<|\"Type\" -> \"EdwardsCurve\", \"CurveName\" -> \"ed25519\", \"PrivateHexString\" -> "
      f"\"{ED_SEED1}\"|>][\"PublicHexString\"]", "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"),
    T(f"PrivateKey[<|\"Type\" -> \"EdwardsCurve\", \"PrivateHexString\" -> \"{ED_SEED2}\"|>][\"PublicHexString\"]",
      "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"),
    ("PublicKey[<|\"Type\" -> \"EllipticCurve\", \"CurveName\" -> \"secp256k1\", \"PublicHexString\" -> "
     "\"02f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9\"|>][\"PublicCurvePoint\"][[2]]",
     str(0x388f7b0f632de8140fe337e62a37f3566500a99934c2231b6cb9fd7584b8e672)),
    ("PublicKey[<|\"Type\" -> \"EllipticCurve\", \"CurveName\" -> \"secp256k1\", \"PublicCurvePoint\" -> {1, 2}|>]",
     Raises(ValueError)),
    ("PublicKey[<|\"Type\" -> \"RSA\", \"PublicModulus\" -> 3233|>][\"PublicExponent\"]", "65537"),
    ("PublicKey[5]", Raises(TypeError)),
    ("Head[PublicKey[<|\"Type\" -> \"RSA\", \"PublicModulus\" -> 3233, \"PublicExponent\" -> 17|>]]", "PublicKey"),
    # RSA encryption round trips
    ("w13rk = GenerateAsymmetricKeyPair[Method -> <|\"KeySize\" -> 1024|>]; "
     "Decrypt[w13rk[\"PrivateKey\"], Encrypt[w13rk[\"PublicKey\"], \"rsa msg\"]]", "\"rsa msg\""),
    ("Decrypt[w13rk[\"PublicKey\"], Encrypt[w13rk[\"PrivateKey\"], {1, 2}]]", "{1, 2}"),
    ("Decrypt[w13rk[\"PrivateKey\"], Encrypt[w13rk[\"PublicKey\"], \"oaep\", Method -> <|\"Padding\" -> \"OAEP\"|>]]",
     "\"oaep\""),
    ("Encrypt[w13rk[\"PublicKey\"], StringJoin[Table[\"x\", 200]]]", Raises(ValueError)),
    ("Encrypt[GenerateAsymmetricKeyPair[\"EllipticCurve\"][\"PublicKey\"], \"x\"]", Raises(ValueError)),
    ("$CryptographicEllipticCurveNames",
     "{\"ed25519\", \"prime192v1\", \"prime256v1\", \"secp192r1\", \"secp224r1\", \"secp256k1\", \"secp256r1\", "
     "\"secp384r1\", \"secp521r1\"}"),
    ("MemberQ[$CryptographicEllipticCurveNames, \"secp256k1\"]", "True"),
    ("Length[$CryptographicEllipticCurveNames]", "9"),
]

SIGNATURES = [
    # RFC 6979 A.2.5 (P-256, SHA-256, message "sample")
    BA(f"GenerateDigitalSignature[\"sample\", {P256_KEY}][\"R\"]",
       "efd48b2aacb6a8fd1140dd9cd45e81d69d2c877b56aaf991c34d0ea84eaf3716"),
    BA(f"GenerateDigitalSignature[\"sample\", {P256_KEY}][\"S\"]",
       "f7cb1c942d657c41d436c7a1b6e29f65f3e900dbb9aff4064dc4ab2f843acda8"),
    (f"VerifyDigitalSignature[{{\"sample\", GenerateDigitalSignature[\"sample\", {P256_KEY}]}}, {P256_KEY}]", "True"),
    # RFC 8032 7.1 tests 1 and 2
    BA(f"GenerateDigitalSignature[ByteArray[{{}}], PrivateKey[<|\"Type\" -> \"EdwardsCurve\", "
       f"\"PrivateHexString\" -> \"{ED_SEED1}\"|>]][\"R\"]",
       "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"),
    BA(f"GenerateDigitalSignature[ByteArray[{{114}}], PrivateKey[<|\"Type\" -> \"EdwardsCurve\", "
       f"\"PrivateHexString\" -> \"{ED_SEED2}\"|>]][\"S\"]",
       "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
    BA(f"GenerateDigitalSignature[ByteArray[{{}}], PrivateKey[<|\"Type\" -> \"EdwardsCurve\", "
       f"\"PrivateHexString\" -> \"{ED_SEED1}\"|>]][\"S\"]",
       "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
    # BIP-340 test vector 0
    BA("GenerateDigitalSignature[ByteArray[ConstantArray[0, 32]], PrivateKey[<|\"Type\" -> \"EllipticCurve\", "
       "\"CurveName\" -> \"secp256k1\", \"PrivateMultiplier\" -> 3|>], Method -> <|\"SignatureType\" -> \"Schnorr\", "
       "\"HashingMethod\" -> None|>][\"R\"]", "e907831f80848d1069a5371b402410364bdf1c5f8307b0084c55f1ce2dca8215"),
    BA("GenerateDigitalSignature[ByteArray[ConstantArray[0, 32]], PrivateKey[<|\"Type\" -> \"EllipticCurve\", "
       "\"CurveName\" -> \"secp256k1\", \"PrivateMultiplier\" -> 3|>], Method -> <|\"SignatureType\" -> \"Schnorr\", "
       "\"HashingMethod\" -> None|>][\"S\"]", "25f66a4a85ea8b71e482a74f382d2ce5ebeee8fdb2172f477df4900d310536c0"),
    ("w13rs = GenerateAsymmetricKeyPair[Method -> <|\"KeySize\" -> 1024|>]; w13sig = GenerateDigitalSignature[\"m\", "
     "w13rs[\"PrivateKey\"]]; VerifyDigitalSignature[{\"m\", w13sig}, w13rs[\"PublicKey\"]]", "True"),
    ("VerifyDigitalSignature[{\"m2\", w13sig}, w13rs[\"PublicKey\"]]", "False"),
    ("VerifyDigitalSignature[{{\"m\", w13sig}, {\"x\", w13sig}}, w13rs[\"PrivateKey\"]]", "{True, False}"),
    ("VerifyDigitalSignature[w13rs[\"PublicKey\"]][{\"m\", w13sig}]", "True"),
    ("w13sig[\"HashingMethod\"]", "\"SHA256\""),
    ("w13sig[\"Type\"]", "\"RSA\""),
    ("GenerateDigitalSignature[\"m\", w13rs[\"PrivateKey\"], Method -> <|\"HashingMethod\" -> \"SHA512\"|>]"
     "[\"HashingMethod\"]", "\"SHA512\""),
    ("GenerateDigitalSignature[\"m\", w13rs[\"PrivateKey\"], Method -> <|\"Padding\" -> \"X931\"|>]",
     Raises(NotImplementedError)),
    ("GenerateDigitalSignature[\"m\", w13rs[\"PublicKey\"]]", Raises(TypeError)),
    ("w13ek = GenerateAsymmetricKeyPair[\"EllipticCurve\"]; VerifyDigitalSignature[{\"z\", "
     "GenerateDigitalSignature[w13ek[\"PrivateKey\"]][\"z\"]}, w13ek[\"PublicKey\"]]", "True"),
    ("w13ed = GenerateAsymmetricKeyPair[\"EdwardsCurve\"]; VerifyDigitalSignature[{{1, 2}, "
     "GenerateDigitalSignature[{1, 2}, w13ed[\"PrivateKey\"]]}, w13ed[\"PublicKey\"]]", "True"),
    ("VerifyDigitalSignature[{\"z\", GenerateDigitalSignature[\"z\", w13ek[\"PrivateKey\"], Method -> "
     "<|\"SignatureType\" -> \"NonDeterministic\"|>]}, w13ek[\"PublicKey\"]]", "True"),
    ("VerifyDigitalSignature[{\"z\", GenerateDigitalSignature[\"z\", w13ek[\"PrivateKey\"]]}, w13ed[\"PublicKey\"]]",
     "False"),
    ("VerifyDigitalSignature[{\"z\", 5}, w13ek[\"PublicKey\"]]", Raises(TypeError)),
    (f"VerifyDigitalSignature[{{\"x\", DigitalSignature[<|\"Type\" -> \"EllipticCurve\", \"CurveName\" -> "
     f"\"secp256r1\", \"HashingMethod\" -> \"SHA256\", \"R\" -> {hexs(SIG_R)}, \"S\" -> {hexs(SIG_S)}|>]}}, "
     f"Import[{F('ecpub.pem')}]]", "False"),
    ("GenerateDigitalSignature[\"m\", w13ek[\"PrivateKey\"]][\"Properties\"]",
     "{\"CurveName\", \"DER\", \"HashingMethod\", \"R\", \"RecoveryParameter\", \"S\", \"SignatureType\", \"Type\"}"),
    P("GenerateDigitalSignature[\"m\", w13ek[\"PrivateKey\"]][\"DER\"]",
      lambda g: base64.b64decode(g.args[0])[0] == 0x30, "DER sequence"),
    ("DigitalSignature[<|\"Type\" -> \"RSA\", \"Signature\" -> ByteArray[{1}]|>][\"Signature\"]", "ByteArray[{1}]"),
    ("DigitalSignature[<|\"Bogus\" -> 1|>]", Raises(ValueError)),
    ("DigitalSignature[3]", Raises(TypeError)),
    (f"w13fsk = Import[{F('ec1.pem')}]; w13fs = GenerateFileSignature[{F('signed.txt')}, w13fsk]; "
     f"VerifyFileSignature[{{{F('signed.txt')}, w13fs}}, w13fsk]", "True"),
    (f"VerifyFileSignature[{{{F('data.csv')}, w13fs}}, w13fsk]", "False"),
    (f"VerifyFileSignature[{{{F('signed.txt')}, DigitalSignature[<|\"Type\" -> \"EllipticCurve\", \"CurveName\" -> "
     f"\"secp256r1\", \"HashingMethod\" -> \"SHA256\", \"R\" -> {hexs(SIG_R)}, \"S\" -> {hexs(SIG_S)}|>]}}, "
     f"Import[{F('ecpub.pem')}]]", "True"),
    (f"GenerateFileSignature[{F('nonexistent.txt')}, w13fsk]", Raises(ValueError)),
    (f"Head[GenerateFileSignature[w13fsk][{F('signed.txt')}]]", "DigitalSignature"),
    (f"VerifyFileSignature[{{{F('nonexistent.txt')}, w13fs}}, w13fsk]", Raises(ValueError)),
]

PEM = [
    (f"Import[{F('rsa8.pem')}][\"PublicKeySize\"]", "Quantity[1024, \"Bits\"]"),
    (f"Import[{F('rsa8.pem')}][\"PublicExponent\"]", "65537"),
    (f"Import[{F('ec1.pem')}][\"CurveName\"]", "\"secp256r1\""),
    (f"Import[{F('ec1.pem')}][\"PublicHexString\"] === Import[{F('ecpub.pem')}][\"PublicHexString\"]", "True"),
    (f"Head[Import[{F('rsacert.pem')}]]", "SecurityCertificate"),
    (f"Import[{F('rsacert.pem')}][\"Subject\"]",
     "<|\"CommonName\" -> \"W13 Test\", \"OrganizationName\" -> \"MikoshiLang\", \"CountryName\" -> \"GB\"|>"),
    (f"Import[{F('rsacert.pem')}][\"SerialNumber\"]", "4660"),
    (f"Import[{F('rsacert.pem')}][\"Version\"]", "3"),
    (f"Import[{F('rsacert.pem')}][\"SignatureAlgorithm\"]", "\"sha256WithRSAEncryption\""),
    (f"Import[{F('rsacert.pem')}][\"ValidityStartDate\"]", "DateObject[{2026, 10, 31, 18, 46, 31}, TimeZone -> 0]"),
    (f"Import[{F('rsacert.pem')}][\"ValidityEndDate\"]", "DateObject[{2126, 10, 7, 18, 46, 31}, TimeZone -> 0]"),
    (f"w13c = Import[{F('rsacert.pem')}]; VerifyDigitalSignature[{{w13c[\"SignedData\"], w13c[\"DigitalSignature\"]}}, "
     f"w13c[\"PublicKey\"]]", "True"),
    (f"w13c2 = Import[{F('eccert.pem')}]; VerifyDigitalSignature[{{w13c2[\"SignedData\"], "
     f"w13c2[\"DigitalSignature\"]}}, w13c2]", "True"),
    (f"Import[{F('eccert.pem')}][\"SignatureAlgorithm\"]", "\"ecdsa-with-SHA384\""),
    (f"Keys[Import[{F('rsacert.pem')}][\"Extensions\"]]",
     "{\"SubjectKeyIdentifier\", \"AuthorityKeyIdentifier\", \"BasicConstraints\"}"),
    (f"SecurityCertificate[Import[{F('rsacert.pem')}, \"Text\"] <> \"\\n\"][\"SerialNumber\"]", "4660"),
    (f"SecurityCertificate[Import[{F('rsacert.pem')}][\"Certificate\"]] === Import[{F('rsacert.pem')}]", "True"),
    ("SecurityCertificate[\"no certificate\"]", Raises(ValueError)),
    ("SecurityCertificate[5]", Raises(TypeError)),
    (f"Import[{F('rsacert.pem')}, \"PublicKey\"]", "Missing[\"NotAvailable\"]"),
    (f"Head[Import[{F('rsa8.pem')}, \"PrivateKey\"]]", "PrivateKey"),
    (f"Export[{F('o1.pem')}, Import[{F('rsa8.pem')}]]; Import[{F('o1.pem')}] === Import[{F('rsa8.pem')}]", "True"),
    (f"Export[{F('o2.pem')}, Import[{F('ec1.pem')}]]; Import[{F('o2.pem')}] === Import[{F('ec1.pem')}]", "True"),
    (f"Export[{F('o3.pem')}, Import[{F('rsacert.pem')}]]; Import[{F('o3.pem')}, \"Text\"] === "
     f"Import[{F('rsacert.pem')}, \"Text\"]", "True"),
    ("w13edk = GenerateAsymmetricKeyPair[\"EdwardsCurve\"][\"PrivateKey\"]; "
     "ImportString[ExportString[w13edk, \"PEM\"], \"PEM\"] === w13edk", "True"),
    ("ExportString[5, \"PEM\"]", Raises(TypeError)),
    ("ImportString[\"-----BEGIN FOO-----\\nAAEC\\n-----END FOO-----\", \"PEM\"][\"Type\"]", "\"FOO\""),
    ("Head[ImportString[\"-----BEGIN FOO-----\\nAAEC\\n-----END FOO-----\", \"PEM\"]]", "ImportedObject"),
    ("ImportString[\"-----BEGIN FOO-----\\nAAEC\\n-----END FOO-----\", \"PEM\"][\"OriginalFormat\"]", "\"PEM\""),
    ("ImportString[\"nothing here\", \"PEM\"]", Raises(ValueError)),
]

# CONVENTIONS.md: symbolic input stays unevaluated, one Raises(Class) per documented error class, every supported
# option value plus a Raises for an unsupported one, and the Check: methods (round trips / self-consistency).
CONVENTIONS = [
    # symbolic numeric arguments stay unevaluated
    ("GenerateAsymmetricKeyPair[Method -> <|\"KeySize\" -> w13n|>]",
     "GenerateAsymmetricKeyPair[Method -> <|\"KeySize\" -> w13n|>]"),
    ("PrivateKey[<|\"Type\" -> \"RSA\", \"PublicModulus\" -> w13m, \"PrivateExponent\" -> 3|>]",
     "PrivateKey[<|\"Type\" -> \"RSA\", \"PublicModulus\" -> w13m, \"PrivateExponent\" -> 3|>]"),
    ("PublicKey[<|\"Type\" -> \"RSA\", \"PublicModulus\" -> w13m|>]", "PublicKey[<|\"Type\" -> \"RSA\", \"PublicModulus\" -> w13m|>]"),
    ("GenerateDerivedKey[\"pw\", Method -> <|\"N\" -> w13n|>]", "GenerateDerivedKey[\"pw\", Method -> <|\"N\" -> w13n|>]"),
    ("ExportByteArray[{1, w13b}, \"Byte\"]", "ExportByteArray[{1, w13b}, \"Byte\"]"),
    ("ExportString[{1, w13b}, \"Real64\"]", "ExportString[{1, w13b}, \"Real64\"]"),
    # wrong kinds of argument: TypeError
    ("GenerateDerivedKey[\"pw\", Method -> <|\"N\" -> \"a\"|>]", Raises(TypeError)),
    ("GenerateDerivedKey[\"pw\", Method -> <|\"N\" -> 2.5|>]", Raises(ValueError)),
    ("GenerateSymmetricKey[5]", Raises(TypeError)),
    ("GenerateSymmetricKey[{1, 2}]", Raises(ValueError)),
    ("Encrypt[GenerateSymmetricKey[], \"x\", Method -> 5]", Raises(TypeError)),
    ("Decrypt[5, Encrypt[\"pw\", \"x\"]]", Raises(TypeError)),
    ("GenerateAsymmetricKeyPair[5]", Raises(ValueError)),
    ("GenerateAsymmetricKeyPair[Method -> <|\"Type\" -> \"EllipticCurve\", \"Compressed\" -> 1|>]", Raises(TypeError)),
    ("GenerateDigitalSignature[\"m\", 5]", Raises(TypeError)),
    ("VerifyDigitalSignature[{\"m\", 5}, 5]", Raises(TypeError)),
    ("VerifyDigitalSignature[{\"m\", DigitalSignature[<|\"Type\" -> \"RSA\", \"Signature\" -> ByteArray[{1}]|>]}, 5]",
     Raises(TypeError)),
    ("GenerateFileSignature[5, GenerateAsymmetricKeyPair[\"EdwardsCurve\"][\"PrivateKey\"]]", Raises(TypeError)),
    ("VerifyFileSignature[{5, DigitalSignature[<|\"Type\" -> \"RSA\", \"Signature\" -> ByteArray[{1}]|>]}, "
     "GenerateAsymmetricKeyPair[\"EdwardsCurve\"][\"PublicKey\"]]", Raises(TypeError)),
    ("EncryptFile[\"pw\", 5]", Raises(TypeError)),
    ("DecryptFile[\"pw\", 5]", Raises(TypeError)),
    ("EncryptedObject[5]", Raises(ValueError)),
    ("DerivedKey[5]", Raises(TypeError)),
    ("VerifyDerivedKey[DerivedKey[\"$scrypt$ln=10,r=8,p=16$TmFDbA$/bq+HJ00cgA\"], 5]", Raises(TypeError)),
    ("SymmetricKey[<|\"Cipher\" -> \"DES\", \"Key\" -> ByteArray[{1, 2, 3, 4, 5, 6, 7, 8}]|>]",
     Raises(NotImplementedError)),
    ("SymmetricKey[<|\"Cipher\" -> \"Nope\", \"Key\" -> ByteArray[{1}]|>]", Raises(ValueError)),
    ("PrivateKey[<|\"Type\" -> \"EdwardsCurve\", \"CurveName\" -> \"ed448\", \"PrivateHexString\" -> \"00\"|>]",
     Raises(NotImplementedError)),
    ("PublicKey[<|\"Type\" -> \"EdwardsCurve\", \"CurveName\" -> \"ed448\", \"PublicHexString\" -> \"00\"|>]",
     Raises(NotImplementedError)),
    ("PublicKey[<|\"Type\" -> \"EllipticCurve\", \"CurveName\" -> \"secp256k1\", \"PublicHexString\" -> \"zz\"|>]",
     Raises(TypeError)),
    ("PublicKey[<|\"Type\" -> \"EllipticCurve\", \"CurveName\" -> \"secp256k1\", \"PublicHexString\" -> \"0201\"|>]",
     Raises(ValueError)),
    ("DigitalSignature[<|\"Type\" -> \"RSA\", \"Signature\" -> ByteArray[{1}]|>][\"Nope\"]", "Missing[\"NotAvailable\"]"),
    ("SecurityCertificate[ByteArray[{48, 3, 2, 1, 0}]]", Raises(ValueError)),
    ("BinarySerialize[x, PerformanceGoal -> \"Speed\"] === BinarySerialize[x]", "True"),
    ("BinarySerialize[x, PerformanceGoal -> \"Bogus\"]", Raises(ValueError)),
    ("BinarySerialize[x, 1]", Raises(ValueError)),
    ("BinaryDeserialize[\"8:\"]", Raises(TypeError)),
    ("Compress[x, Method -> \"zlib\"]", Raises(NotImplementedError)),
    ("Uncompress[5]", Raises(ValueError)),
    ("Import[5]", Raises(TypeError)),
    ("ImportString[5, \"CSV\"]", Raises(TypeError)),
    ("ImportString[\"1\", \"CSV\", Bogus -> 1]", Raises(ValueError)),
    ("ImportByteArray[ByteArray[{1}], \"NoSuchFormat\"]", Raises(ValueError)),
    ("Export[5, {1}]", Raises(TypeError)),
    ("ExportString[{1}, 5]", Raises(TypeError)),
    ("ExportByteArray[{1}, 5]", Raises(TypeError)),
    ("ExportString[{1}, \"MX\", MetaInformation -> {}]", Raises(NotImplementedError)),
    ("FileFormatProperties[5]", Raises(TypeError)),
    ("FileFormatProperties[\"CSV\", 5]", Raises(TypeError)),
    ("SetFileFormatProperties[\"TSV\", \"MIMETypes\" -> 5]", Raises(TypeError)),
    ("FileNameToFormatList[5]", Raises(TypeError)),
    ("MIMETypeToFormatList[5]", Raises(TypeError)),
    ("FileFormat[5]", Raises(TypeError)),
    ("FileFormatQ[5, \"CSV\"]", Raises(TypeError)),
    ("FileConvert[5]", Raises(TypeError)),
    ("CreateArchive[5, \"x.zip\"]", Raises(TypeError)),
    ("ExtractArchive[5, \"x\"]", Raises(TypeError)),
    ("PersistenceLocation[5]", Raises(TypeError)),
    ("ToMemory[w13tm2]", "w13tm2"),
    # option values
    ("w13ko = GenerateSymmetricKey[]; Table[Decrypt[w13ko, Encrypt[w13ko, \"opt\", Method -> <|\"BlockMode\" -> m|>]], "
     "{m, {\"ECB\", \"CBC\", \"CFB\", \"OFB\", \"CTR\"}}]", "{\"opt\", \"opt\", \"opt\", \"opt\", \"opt\"}"),
    ("Encrypt[GenerateSymmetricKey[], \"x\", Method -> <|\"BlockMode\" -> \"XTS\"|>]", Raises(ValueError)),
    ("Encrypt[GenerateSymmetricKey[], \"x\", Method -> <|\"Bogus\" -> 1|>]", Raises(ValueError)),
    ("Encrypt[GenerateSymmetricKey[], \"x\", Method -> <|\"Padding\" -> \"Bogus\"|>]", Raises(ValueError)),
    ("Encrypt[GenerateSymmetricKey[], \"x\", Method -> <|\"Padding\" -> None|>]", Raises(ValueError)),
    BA("w13kq = GenerateSymmetricKey[]; Decrypt[w13kq, Encrypt[w13kq, ByteArray[Range[16]], "
       "Method -> <|\"Padding\" -> None|>]]", bytes(range(1, 17)).hex()),
    ("Table[GenerateSymmetricKey[Method -> c][\"Cipher\"], {c, {\"AES128\", \"AES192\", \"AES256\", \"Blowfish\", "
     "\"RC4\"}}]", "{\"AES128\", \"AES192\", \"AES256\", \"Blowfish\", \"RC4\"}"),
    ("Table[GenerateSymmetricKey[Method -> c], {c, {\"CAST5\", \"IDEA\"}}]", Raises(NotImplementedError)),
    ("GenerateSymmetricKey[Method -> <|\"Cipher\" -> \"AES128\", \"KeySize\" -> 192|>]", Raises(ValueError)),
    ("w13sk = Import[" + F("ec1.pem") + "]; Table[VerifyDigitalSignature[{\"h\", GenerateDigitalSignature[\"h\", w13sk, "
     "Method -> <|\"HashingMethod\" -> h|>]}, w13sk], {h, {\"SHA1\", \"SHA256\", \"SHA384\", \"SHA512\", \"SHA3-256\", "
     "\"MD5\", \"RIPEMD160\", \"Keccak256\"}}]", "{True, True, True, True, True, True, True, True}"),
    ("GenerateDigitalSignature[\"h\", w13sk, Method -> <|\"HashingMethod\" -> \"Bogus\"|>]", Raises(ValueError)),
    ("GenerateDigitalSignature[\"h\", w13sk, Method -> <|\"SignatureType\" -> \"Bogus\"|>]", Raises(ValueError)),
    ("GenerateDigitalSignature[\"h\", w13sk, Method -> <|\"SignatureType\" -> \"Schnorr\"|>]", Raises(ValueError)),
    ("GenerateDigitalSignature[\"h\", w13sk, Method -> <|\"Bogus\" -> 1|>]", Raises(ValueError)),
    ("GenerateDigitalSignature[\"h\", w13sk][\"SignatureType\"]", "\"Deterministic\""),
    ("GenerateDigitalSignature[\"h\", w13sk] === GenerateDigitalSignature[\"h\", w13sk]", "True"),
    ("GenerateDigitalSignature[\"h\", w13sk, Method -> <|\"SignatureType\" -> \"NonDeterministic\"|>][\"SignatureType\"]",
     "\"NonDeterministic\""),
    ("GenerateDigitalSignature[ByteArray[Range[32]], w13sk, Method -> <|\"HashingMethod\" -> None|>][\"HashingMethod\"]",
     "Null"),
    ("GenerateDigitalSignature[\"not hex\", w13sk, Method -> <|\"HashingMethod\" -> None|>]", Raises(ValueError)),
    ("GenerateAsymmetricKeyPair[Method -> <|\"Type\" -> \"RSA\", \"KeySize\" -> 512, \"PublicExponent\" -> 3|>]"
     "[\"PublicKey\"][\"PublicExponent\"]", "3"),
    ("GenerateAsymmetricKeyPair[Method -> <|\"Type\" -> \"RSA\", \"KeySize\" -> 512, \"PublicExponent\" -> 4|>]",
     Raises(ValueError)),
    ("GenerateAsymmetricKeyPair[Method -> <|\"Type\" -> \"RSA\", \"Bogus\" -> 1|>]", Raises(ValueError)),
    ("Table[GenerateAsymmetricKeyPair[Method -> <|\"Type\" -> \"EllipticCurve\", \"CurveName\" -> c|>][\"PublicKey\"]"
     "[\"CurveName\"], {c, {\"prime192v1\", \"secp224r1\", \"prime256v1\", \"secp384r1\", \"secp521r1\"}}]",
     "{\"secp192r1\", \"secp224r1\", \"secp256r1\", \"secp384r1\", \"secp521r1\"}"),
    ("GenerateDerivedKey[\"pw\", Method -> <|\"Function\" -> \"bcrypt\"|>]", Raises(ValueError)),
    ("GenerateDerivedKey[\"pw\", Method -> <|\"Bogus\" -> 1|>]", Raises(ValueError)),
    (f"EncryptFile[\"pw\", {F('signed.txt')}, {F('of.bin')}]; EncryptFile[\"pw\", {F('signed.txt')}, {F('of.bin')}, "
     f"OverwriteTarget -> True]", f"{F('of.bin')}"),
    (f"EncryptFile[\"pw\", {F('signed.txt')}, {F('of.bin')}, OverwriteTarget -> 1]", Raises(TypeError)),
    (f"EncryptFile[\"pw\", {F('signed.txt')}, {F('of.bin')}, Bogus -> 1]", Raises(ValueError)),
    # Check: round trips and self-consistency
    ("w13rt = GenerateAsymmetricKeyPair[\"EdwardsCurve\"]; PublicKey[<|\"Type\" -> \"EdwardsCurve\", \"PublicHexString\" -> "
     "w13rt[\"PrivateKey\"][\"PublicHexString\"]|>] === w13rt[\"PublicKey\"]", "True"),
    ("w13rt2 = GenerateAsymmetricKeyPair[\"Bitcoin\"]; PublicKey[<|\"Type\" -> \"EllipticCurve\", \"CurveName\" -> "
     "\"secp256k1\", \"PublicByteArray\" -> w13rt2[\"PublicKey\"][\"PublicByteArray\"]|>][\"PublicCurvePoint\"] === "
     "w13rt2[\"PublicKey\"][\"PublicCurvePoint\"]", "True"),
    ("w13rt3 = GenerateAsymmetricKeyPair[Method -> <|\"KeySize\" -> 512|>][\"PrivateKey\"]; "
     "PowerMod[PowerMod[42, w13rt3[\"PublicExponent\"], w13rt3[\"PublicModulus\"]], w13rt3[\"PrivateExponent\"], "
     "w13rt3[\"PublicModulus\"]] == 42", "True"),
    ("w13dt = GenerateDerivedKey[\"pw\", Method -> <|\"N\" -> 64|>]; VerifyDerivedKey[DerivedKey[w13dt[\"PHCString\"]], \"pw\"]",
     "True"),
    ("Table[Uncompress[Compress[e]] === e, {e, {{1, 2.5, \"s\"}, w13u[x, y], 2^200, 1/3, <|a -> b|>}}]",
     "{True, True, True, True, True}"),
    ("Table[BinaryDeserialize[BinarySerialize[e]] === e, {e, {{1, 2.5, \"s\"}, w13u[x, y], 2^200, 1/3, <|a -> b|>}}]",
     "{True, True, True, True, True}"),
    ("Table[ImportString[ExportString[{{1, 2}, {3, 4}}, f], f], {f, {\"CSV\", \"TSV\", \"JSON\", \"Table\"}}]",
     "{{{1, 2}, {3, 4}}, {{1, 2}, {3, 4}}, {{1, 2}, {3, 4}}, {{1, 2}, {3, 4}}}"),
]


def test_curve_parameters():
    from mikoshilang.library import wl_crypto as C
    for c in C.CURVES.values():
        assert c.on_curve(c.G) and c.mul(c.n) is None and C._probable_prime(c.n) and C._probable_prime(c.p)


def test_primitive_vectors():
    import hashlib
    from mikoshilang.library import wl_crypto as C
    assert C.keccak(b"", 256).hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    assert C._ripemd160_py(b"abc").hex() == "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc"
    assert C.digest("SHA3-256", b"abc") == hashlib.sha3_256(b"abc").digest()
    for k in (16, 24, 32):
        a = C.AES(bytes(range(k)))
        assert a.decrypt_block(a.encrypt_block(b"0123456789abcdef")) == b"0123456789abcdef"
    sig = C.ed25519_sign(bytes.fromhex(ED_SEED1), b"")
    assert (sig[0] + sig[1]).hex().startswith("e5564300c360ac72")


CASES = (SERIALISATION + FORMATS + REGISTRY + FILES + MISC + HASHES_AND_CIPHERS + SYMMETRIC + DERIVED + ASYMMETRIC
         + SIGNATURES + PEM + CONVENTIONS)


@pytest.mark.parametrize("case", CASES, ids=[c[0].replace(D, "TMP")[:80] for c in CASES])
def test_wolfram_w13(case):
    check(case)
