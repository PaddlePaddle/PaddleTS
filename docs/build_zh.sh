rm -rf build
sphinx-build -b gettext . build/gettext
sphinx-intl update -p ./build/gettext -l zh_CN
sphinx-build -b html -D language=zh_CN . build/html/zh_CN
