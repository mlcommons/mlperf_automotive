import markdown
import unittest
from swaggermarkdown.swaggermarkdown import SwaggerExtension

style = '''
<style>
  body {
    font-size: 16px;
    font-family: sans-serif;
  }
  [data-type="sw-table"] {
    border-collapse: collapse;
    min-width: 400px;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    margin-bottom: 40px;
  }
  th,
  td {
      padding: 12px 15px;
  }

  tbody tr {
    border-bottom: 1px solid #ddd;
  }

  th {
    border-bottom: 2px solid #ccc;
  }

  tbody tr:nth-of-type(even) {
      background-color: #f3f3f3;
  }

  caption {
    padding-bottom: 6px;
  }

  .sw-label {
    color: #666;
    display: inline-block;
    min-width: 80px;
    text-align: right;
  }

  .sw-verb {
    padding: 4px 8px;
    background-color: #0366d6;
    border-radius: 4px;
    color: #fff;
  }

  .sw-path-url {
      background: #eee;
      padding: 4px 8px;
      border-radius: 4px;
  }
</style>
'''

class TestSwaggerExtension(unittest.TestCase):

    def test_instanciate_extension(self):
        text = 'some text'
        self.assertEqual('<p>'+text+'</p>', markdown.markdown(text, extensions=['swaggermarkdown']))
    
    def test_instanciate_extension_with_file(self):
        md = markdown.Markdown(extensions=[SwaggerExtension(file='tests/test_swagger.json')])
        text = 'some text'
        self.assertEqual('<p>'+text+'</p>', md.convert(text))

    def test_swagger_include(self):
        md = markdown.Markdown(extensions=[SwaggerExtension(file='tests/test_swagger.json')])
        text = ':swg-def: FirstDefinition'
        converted = md.convert(text)
        self.assertTrue(converted.startswith('<table '))

    def test_generate_file(self):
        md = markdown.Markdown(extensions=[SwaggerExtension(file='tests/test_swagger.json')])
        with open('tests/test.md', 'r') as md_input:
            html = md.convert(md_input.read())
            with open("tests/test.html", "w") as out:
              out.write(style)
              out.write(html)

    def test_generate_pet_file(self):
        md = markdown.Markdown(extensions=[SwaggerExtension(
          file='tests/pet_store.json',
          definitionsUrlRoot='/types'
        )])
        with open('tests/pet.md', 'r') as md_input:
            html = md.convert(md_input.read())
            self.assertIn('<a href="/types#/definitions/Category">Category</a>', html)
            with open("tests/pet.html", "w") as out:
              out.write(style)
              out.write(html)

    def test_swagger_explicit_file_include(self):
        md = markdown.Markdown(extensions=[SwaggerExtension()])
        text = ':swg-def: tests/test_swagger.json FirstDefinition'
        converted = md.convert(text)
        self.assertTrue(converted.startswith('<table '))

    def test_definition_data(self):
        md = markdown.Markdown(extensions=[SwaggerExtension()])
        text = ':swg-def: tests/test_swagger.json FirstDefinition'
        converted = md.convert(text)
        self.assertIn('Some example text', converted)
        self.assertIn('example', converted)
        self.assertIn('2050', converted)
        self.assertIn('1999', converted)
        self.assertIn('arrayOfStrings', converted)
        self.assertIn('minItems', converted)
        self.assertIn('123', converted)
        self.assertIn('hello world', converted)
        self.assertIn('<strong>name</strong>', converted)
        self.assertIn('array of string', converted)
        self.assertIn('<a href="#/definitions/SecondDefinition">SecondDefinition</a>', converted)
        self.assertIn('integer int32', converted)
        self.assertIn('arrayOfObject</strong>[0].id', converted)
        self.assertIn('array of <a href="#/definitions/SecondDefinition">SecondDefinition', converted)

    def test_enum_definition(self):
        md = markdown.Markdown(extensions=[SwaggerExtension()])
        text = ':swg-def: tests/test_swagger.json Enum'
        converted = md.convert(text)
        self.assertIn('enum1', converted)
        self.assertIn('Enum description', converted)

    def test_array_definition(self):
        md = markdown.Markdown(extensions=[SwaggerExtension()])
        text = ':swg-def: tests/test_swagger.json ArrayOfFriends'
        converted = md.convert(text)
        self.assertIn('array of <a href="#/definitions/Friends">', converted)

    def test_path_data(self):
        md = markdown.Markdown(extensions=[SwaggerExtension()])
        text = ':swg-path: tests/test_swagger.json /my-project'
        converted = md.convert(text)
        self.assertIn('<td>array of <a href="#/definitions/Friends">Friends</a></td>', converted)
        self.assertIn('curl -i https://example.com/my-project', converted)

    def test_second_definition(self):
        md = markdown.Markdown(extensions=[SwaggerExtension(file='tests/test_swagger.json')])
        text = '''
:swg-def: tests/test_swagger.json FirstDefinition
:swg-def: SecondDefinition
'''
        converted = md.convert(text)
        self.assertIn('Some example text', converted)
        self.assertIn('another def', converted)

    def test_definitionsUrlRoot(self):
        md = markdown.Markdown(extensions=[SwaggerExtension(definitionsUrlRoot='/types')])
        text = ':swg-def: tests/test_swagger.json FirstDefinition'
        converted = md.convert(text)
        self.assertIn('<a href="/types#/definitions/SecondDefinition">SecondDefinition</a>', converted)


if __name__ == '__main__':
    unittest.main()