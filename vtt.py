import re

def extract_speaker_text(vtt_path, first_name, last_name):
    """
    Extracts all text spoken by a specific speaker from a VTT file.

    :param vtt_path: Path to the VTT file.
    :param first_name: First name of the speaker.
    :param last_name: Last name of the speaker.
    :return: List of text segments spoken by the speaker.
    """
    # Construct the speaker tag as it appears in the VTT file
    speaker_tag = f"<v {last_name}, {first_name}>"
    
    # Compile a regex pattern to match the speaker's text
    # This pattern looks for lines like: <v Merklein, Kai>Text</v>
    pattern = re.compile(re.escape(speaker_tag) + r'(.*?)</v>', re.IGNORECASE)
    
    extracted_texts = []

    print('Texts found by speaker:', speaker_tag)
    
    # Open and read the VTT file
    with open(vtt_path, 'r', encoding='utf-8') as file:
        segment = ""
        for line in file:
            # concat the line to segment, until the line contain '</v>'
            if '</v>' in line:
                segment += line
                match = pattern.search(segment)
                if match:
                    # If a match is found, extract and clean the text
                    text = match.group(1)
                    extracted_texts.append(text )
                    segment = ""
            else:
                # add line to segment, but leading and trailing whitespaces are removed
                segment += line.strip() + ' '
    
    return extracted_texts

# Example usage:
if __name__ == "__main__":
    # Path to your VTT file
    vtt_file_path = '/Users/D046675/Downloads/Walk-Through and Discussion of ACD for DSSA.vtt'
    
    # Extract texts spoken by Kai Merklein
    kai_texts = extract_speaker_text(vtt_file_path, first_name="Kai", last_name="Merklein")
    
    # Print the extracted texts
    print("Texts spoken by Kai Merklein:")
    for idx, text in enumerate(kai_texts, 1):
        # print(f"{idx}. {text}")
        print(text)
